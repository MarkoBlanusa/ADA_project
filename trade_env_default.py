import gymnasium.utils.seeding
import numpy as np
import gymnasium
from gymnasium import spaces
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import pandas as pd
import mplfinance as mpf
from sklearn.preprocessing import MinMaxScaler
import logging
from torch import Tensor
import torch
import ray
from ray.data import Dataset

logging.basicConfig(level=logging.DEBUG)


class TradingEnvironment(gymnasium.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        normalized_data: Dataset,
        input_length=96,
        render_mode=None,
    ):
        super(TradingEnvironment, self).__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Trading states
        self.normalized_data = normalized_data
        self.input_length = input_length
        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        self.exit_price = 0
        self.position_size = 0
        self.leverage = 0
        self.leverage_limit = 50
        self.stop_loss = 0
        self.take_profit = 0
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.current_balance = self.balance
        self.market_transaction_fee = 0.0005
        self.limit_transaction_fee = 0.0002
        self.position_open_steps = 0
        self.not_in_position_steps = 0
        self.collect_pnl = False

        self.trades = []
        self.balance_history = []

        # Define meaningful default values for the static elements.
        self.default_static_values = np.array(
            [
                self.initial_balance,  # Initial balance.
                0,  # Initial position size.
                0,  # Initial leverage.
                0,  # Initial stop loss.
                0,  # Initial take profit.
                0,  # Initial entry price.
                0,  # Initial exit price.
                0,  # Initial trade action
                0,  # Initial position proportion action
                0,  # Initial stop loss adj action
                0,  # Initial take profit adj action
                0,  # Initial stop loss pct action
                0,  # Initial take profit pct action
                0,  # Initial reward
            ]
        )

        # Action space definition
        self.action_space = spaces.Dict(
            {
                "trade_action": spaces.Discrete(3),  # 0: hold, 1: buy, 2: sell
                "position_proportion": spaces.Box(
                    low=0.0, high=1.0, shape=(1,), dtype=np.float32
                ),
                "stop_loss_adj": spaces.Box(
                    low=-0.05, high=0.3, shape=(1,), dtype=np.float32
                ),
                "take_profit_adj": spaces.Box(
                    low=-0.05, high=0.3, shape=(1,), dtype=np.float32
                ),
                "stop_loss_pct": spaces.Box(
                    low=0.0005, high=0.3, shape=(1,), dtype=np.float32
                ),
                "take_profit_pct": spaces.Box(
                    low=0.0005, high=0.3, shape=(1,), dtype=np.float32
                ),
            }
        )

        self.sequence_buffer = []
        self.load_initial_sequences()

        self.state = self.sequence_buffer[0]

        print("STATE SEQUENCE SHAPE: ", self.state.shape)

        # print(self.state["sequential"].shape)
        # print(self.state["static"].shape)
        # print(self.state["sequential"])
        # print(self.state["static"])
        # print(type(self.state["sequential"]))
        # print(type(self.state["static"]))

        # num_seq = self.state["sequential"].shape[0]
        seq_features = self.state.shape[1]
        print("SEQ FEATURES OF THE OBSERVATION SPACE: ", seq_features)

        # Observation space definition
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(input_length, seq_features),
            dtype=np.float32,
        )

    def load_initial_sequences(self):
        self.sequence_buffer.clear()
        self.iterator = iter(
            self.normalized_data.iter_batches(
                batch_size=1, prefetch_batches=self.input_length
            )
        )
        normalized_default_static_values = self.normalize_action_states(
            self.default_static_values
        )
        for _ in range(self.input_length):
            try:
                initial_data = next(self.iterator)["data"][0]
                initial_data[:, -14:] = normalized_default_static_values
                self.sequence_buffer.append(initial_data)
            except StopIteration:
                break  # In case there are fewer batches than the input_length

    def update_sequences(self, new_data):
        # new_data should contain the updated last 14 features for the most current observation
        # This method will cascade this new_data through the last 14 features of each sequence's corresponding observation
        normalized_new_data = self.normalize_action_states(new_data)
        # Handle the addition of a completely new sequence and retire the oldest sequence
        try:
            new_sequence = next(self.iterator)["data"][0]
            # Replace the oldest sequence with a new one if available
            self.sequence_buffer.append(new_sequence)
            self.sequence_buffer.pop(0)
            # Update the last row's last 7 features in each sequence appropriately
            for i in range(len(self.sequence_buffer)):
                # Update the (length-i-1)'th observation in the i'th sequence in the buffer
                self.sequence_buffer[i][-i - 1, -14:] = normalized_new_data
            next_state = self.sequence_buffer[0]
            terminated = False
            return next_state, terminated, {}

        except StopIteration:

            # Even if no more sequences are left for the iterator, finish the remaining sequences inside the buffer
            if len(self.sequence_buffer) > 1:
                self.sequence_buffer.pop(0)
                for i in range(len(self.sequence_buffer)):
                    self.sequence_buffer[i][-i - 1, -14:] = normalized_new_data
                next_state = self.sequence_buffer[0]
                terminated = False
                return next_state, terminated, {}

            else:
                # Reset the episode if the batch buffer ends
                next_state, info = self.reset()
                terminated = True
                return next_state, terminated, info

    def step(self, action):

        trade_info = {
            "entry_price": None,
            "exit_price": None,
            "position_size": None,
            "leverage": None,
            "stop_loss": None,
            "take_profit": None,
            "is_long": None,
            "pnl": None,
            "balance": None,
        }

        trade_action = action["trade_action"]
        position_proportion = action["position_proportion"][0]
        stop_loss_adj = action["stop_loss_adj"][0]
        take_profit_adj = action["take_profit_adj"][0]
        stop_loss_pct = action["stop_loss_pct"][0]
        take_profit_pct = action["take_profit_pct"][0]

        # Normal distribution centered slightly below zero
        mean_slippage = -0.0005
        std_dev_slippage = 0.0005
        slippage = np.random.normal(mean_slippage, std_dev_slippage)

        # Ensuring slippage remains within reasonable bounds
        slippage = max(min(slippage, 0.001), -0.001)

        # Extract the last observation of the current state for trading decisions
        current_close = self.state[-1, 3]
        current_high = self.state[-1, 1]
        current_low = self.state[-1, 2]

        # For margin liquidation calculations
        if self.position > 0:
            worst_case_price = current_low
        elif self.position < 0:
            worst_case_price = current_high
        else:
            worst_case_price = 0

        # Calculate adjusted entry price with slippage
        # entry_price = current_close * (1 + slippage) if trade_action != 0 else None

        self.exit_price = 0
        transaction_cost = 0
        pnl = 0
        pnl_percent = 0
        transaction_cost_percent = 0
        self.collect_pnl = False

        # Check first for an open position to close before checking for an entry in order to avoid an instant close after entering the position
        # Determine if stop loss or take profit was hit, or if the position has been liquidated
        if self.position != 0:

            self.position_open_steps += 1

            # Check for position liquidation due to reaching margin call threshold
            unrealized_pnl = (
                (worst_case_price - self.entry_price)
                * self.position_size
                * (1 if self.position > 0 else -1)
            )

            unrealized_pnl_close = (
                (current_close - self.entry_price)
                * self.position_size
                * (1 if self.position > 0 else -1)
            )

            # Update current balance
            self.current_balance = self.balance + unrealized_pnl_close

            if -unrealized_pnl >= 0.95 * self.balance:
                self.exit_price = (
                    worst_case_price  # Assuming immediate liquidation at current price
                )
                pnl = unrealized_pnl  # Final P&L after liquidation
                pnl_percent = pnl / self.balance
                transaction_cost = (
                    self.market_transaction_fee * self.position_size * self.exit_price
                )
                transaction_cost_percent = transaction_cost / self.balance
                self.balance += pnl - transaction_cost
                self.current_balance = self.balance
                self.entry_price = 0
                self.leverage = 0
                self.position = 0
                self.position_open_steps = 0
                self.position_size = 0
                self.stop_loss = 0
                self.take_profit = 0
                self.collect_pnl = True

                # Log exit
                trade_info.update(
                    {"exit_price": self.exit_price, "pnl": pnl, "balance": self.balance}
                )

            # If the position hasn't been liquidated check for a potential SL or TP
            if self.position > 0:  # Long position
                # SL and TP exit
                if current_low <= self.stop_loss and current_high >= self.take_profit:
                    self.exit_price = self.stop_loss * (1 + slippage)
                elif current_low <= self.stop_loss or current_high >= self.take_profit:
                    self.exit_price = (
                        self.stop_loss
                        if current_low <= self.stop_loss
                        else self.take_profit
                    ) * (1 + slippage)

            elif self.position < 0:  # Short position
                if current_high >= self.stop_loss and current_low <= self.take_profit:
                    self.exit_price = self.stop_loss * (1 + slippage)
                elif current_high >= self.stop_loss or current_low <= self.take_profit:
                    self.exit_price = (
                        self.stop_loss
                        if current_high >= self.stop_loss
                        else self.take_profit
                    ) * (1 + slippage)
            if self.exit_price != 0:
                pnl = (
                    (self.exit_price - self.entry_price)
                    * self.position_size
                    * (1 if self.position > 0 else -1)
                )
                pnl_percent = pnl / self.balance
                transaction_cost = (
                    self.limit_transaction_fee * self.position_size * self.exit_price
                )
                transaction_cost_percent = transaction_cost / self.balance
                self.balance += pnl
                self.balance -= transaction_cost
                self.current_balance = self.balance
                self.position = 0  # Reset position
                self.leverage = 0
                self.position_size = 0
                self.entry_price = 0
                self.stop_loss = 0
                self.take_profit = 0
                self.position_open_steps = 0  # Reset position duration counter
                self.collect_pnl = True

                # Log exit
                trade_info.update(
                    {"exit_price": self.exit_price, "pnl": pnl, "balance": self.balance}
                )

                # Adjust stop loss and take profit based on current high/low if nothing made our position to close
                if self.position > 0:
                    self.stop_loss += stop_loss_adj * current_low
                    self.take_profit += take_profit_adj * current_high
                elif self.position < 0:
                    self.stop_loss -= stop_loss_adj * current_high
                    self.take_profit -= take_profit_adj * current_low

        # Check for an entry
        if trade_action != 0 and self.position == 0 and self.balance > 0:
            self.position = 1 if trade_action == 1 else -1
            self.entry_price = current_close * (1 + slippage)
            max_position_size = self.balance * self.leverage_limit / self.entry_price
            self.position_size = max_position_size * position_proportion
            self.leverage = min(
                self.leverage_limit,
                self.position_size * self.entry_price / self.balance,
            )
            self.stop_loss = self.entry_price * (
                1 - stop_loss_pct if self.position > 0 else 1 + stop_loss_pct
            )
            self.take_profit = self.entry_price * (
                1 + take_profit_pct if self.position > 0 else 1 - take_profit_pct
            )
            transaction_cost = (
                self.market_transaction_fee * self.position_size * self.entry_price
            )
            transaction_cost_percent = transaction_cost / self.balance
            self.balance -= transaction_cost
            self.current_balance = self.balance

            # Log entry
            trade_info.update(
                {
                    "entry_price": self.entry_price,
                    "position_size": self.position_size,
                    "leverage": self.leverage,
                    "stop_loss": self.stop_loss,
                    "take_profit": self.take_profit,
                    "is_long": self.position > 0,
                    "balance": self.balance,
                }
            )

        # When no positions are open and we aren't trading
        else:
            self.not_in_position_steps += 1

        # Check for balance depletion to reset faster and penalize

        reset_penalty = self.check_reset_conditions()

        # Make sure the balance stays under the bounds

        self.balance = min(max(self.balance, 0), 1e7)
        self.current_balance = min(max(self.current_balance, 0), 1e7)

        # Start collecting and updating everything related to the end of the step

        self.trades.append(trade_info)
        self.balance_history.append(self.balance)

        self.current_step += 1

        reward = self.calculate_reward(
            action, pnl, pnl_percent, transaction_cost_percent
        )
        reward -= reset_penalty

        # Now update the static part based on the action's results
        updated_static_state = np.array(
            [
                self.current_balance,
                self.position_size,
                self.leverage,
                self.stop_loss,
                self.take_profit,
                self.entry_price,
                self.exit_price,
                trade_action,
                position_proportion,
                stop_loss_adj,
                take_profit_adj,
                stop_loss_pct,
                take_profit_pct,
                reward,
            ]
        )

        next_state, terminated, info = self.update_sequences(updated_static_state)

        print("MAX VALUE NEXT STATE: ", max(next_state[-1]))
        print("MIN VALUE NEXT STATE: ", min(next_state[-1]))

        truncated = False

        # logging.debug(f"State after processing: {self.state}")

        return (
            next_state,
            reward,
            terminated,
            truncated,
            info,
        )

    def check_reset_conditions(self):
        if self.balance < 0.05 * self.initial_balance:
            self.balance = self.initial_balance
            reset_penalty = 20
            return reset_penalty
        else:
            return 0

    def normalize_action_states(self, states):
        # Initialize normalized states array
        normalized_states = np.zeros_like(states)

        balance_min = 0
        balance_max = 1e7
        position_size_min = 0
        position_size_max = 1e7
        leverage_min = 0
        leverage_max = 50
        trade_action_max = 2
        trade_action_min = 0
        stop_adj_max = 0.3001
        stop_adj_min = -0.05001
        profit_adj_max = 0.3001
        profit_adj_min = -0.05001
        reward_max = 1e4
        reward_min = -1e4

        # Normalize each feature accordingly
        normalized_states[0] = (states[0] - balance_min) / (balance_max - balance_min)
        normalized_states[1] = (states[1] - position_size_min) / (
            position_size_max - position_size_min
        )
        normalized_states[2] = (states[2] - leverage_min) / (
            leverage_max - leverage_min
        )
        normalized_states[3] = states[3]  # Already normalized
        normalized_states[4] = states[4]  # Already normalized
        normalized_states[5] = states[5]  # Already normalized
        normalized_states[6] = states[6]  # Already normalized
        normalized_states[7] = (states[7] - trade_action_min) / (
            trade_action_max - trade_action_min
        )
        normalized_states[8] = states[8]  # Already normalized
        normalized_states[9] = (states[9] - stop_adj_min) / (
            stop_adj_max - stop_adj_min
        )
        normalized_states[10] = (states[10] - profit_adj_min) / (
            profit_adj_max - profit_adj_min
        )
        normalized_states[11] = states[11]  # Already normalized
        normalized_states[12] = states[12]  # Already normalized
        normalized_states[13] = (states[13] - reward_min) / (reward_max - reward_min)

        print("NORMALIZED STATES ACTIONS REWARDS : ", normalized_states)

        return normalized_states

    def calculate_reward(self, action, pnl, pnl_percent, transaction_fee):
        # Basic reward adjusted
        basic_reward = pnl_percent - transaction_fee

        if basic_reward < 0 and basic_reward > -1 and self.collect_pnl == True:
            # Calculate the recovery factor: how much gain is needed to recover the loss
            recovery_factor = (
                1 / (1 + basic_reward) - 1
            )  # This calculates the % gain needed to return to the original balance
            basic_reward = -np.log(
                1 + recovery_factor
            )  # Use logarithmic penalty to scale the impact
            # Scaling penalty based on account size, using a logarithmic scale to moderate impact
            account_scale = np.log10(
                self.balance + 10
            )  # Logarithmic scale based on balance
            scaled_penalty = (
                basic_reward * account_scale
            )  # Scale the penalty by the account size log
            basic_reward = scaled_penalty
        else:
            basic_reward = np.tanh(
                basic_reward
            )  # Continue to cap gains between 0 and 1

        # Penalty for illegal actions
        if self.position != 0 and (
            action["trade_action"] != 0 or action["position_proportion"] != 0
        ):
            illegal_penalty = 0.1
        else:
            illegal_penalty = 0

        # Penalty based on the length of time the position is open
        time_penalty = 0.0001 * self.position_open_steps
        # Penalty based on the duration without trading
        nothing_penalty = 0.0001 * self.not_in_position_steps

        # Calculate drawdown as a penalty
        max_balance = max(self.balance, getattr(self, "max_balance", self.balance))
        self.max_balance = max_balance  # Update historical maximum balance
        drawdown = (max_balance - self.balance) / max_balance
        drawdown_penalty = drawdown * 0.01  # Adjust scale to keep impact meaningful

        # Determine acceptable risk percentage based on balance
        def acceptable_risk(balance):
            if balance <= 10000:
                return 0.02  # 2% risk for balances ≤ $10,000
            elif balance <= 100000:
                return 0.01  # 1% risk for balances from $10,001 to $100,000
            elif balance <= 1000000:
                return 0.005  # 0.5% risk for balances from $100,001 to $1,000,000
            else:
                return 0.0025  # 0.25% risk for balances over $1,000,000

        # Risk management reward or penalty
        max_risk_percent = acceptable_risk(self.balance)
        if self.position > 0:
            risk_per_trade = (
                (self.entry_price - self.stop_loss) * self.position_size
            ) / self.balance
        elif self.position < 0:
            risk_per_trade = (
                (self.stop_loss - self.entry_price) * self.position_size
            ) / self.balance
        else:
            risk_per_trade = 0

        # Scale and penalize dynamically based on calculated risk

        risk_penalty = max(0, risk_per_trade - max_risk_percent) * 0.01

        # Reward for trade efficiency in terms of Sharpe ratio-like measure
        if risk_per_trade > 0:  # To avoid division by zero
            trade_efficiency = pnl / (risk_per_trade * self.balance)
        else:
            trade_efficiency = 0

        trade_efficiency_reward = (
            max(0, trade_efficiency - 1) * 0.01
        )  # Only reward if efficiency > 1

        # Combine all components to form the final reward
        return (
            basic_reward
            - time_penalty
            - nothing_penalty
            - illegal_penalty
            - drawdown_penalty
            - risk_penalty
            + trade_efficiency_reward
        )

    def reset(self, seed=None, **kwargs):

        super().reset(seed=seed, **kwargs)  # Call to super to handle seeding properly

        self.trades = []
        self.balance_history = []

        self.current_step = 0
        self.position = 0
        self.entry_price = 0
        self.exit_price = 0
        self.position_size = 0
        self.leverage = 0
        self.leverage_limit = 50
        self.stop_loss = 0
        self.take_profit = 0
        self.initial_balance = 10000
        self.balance = self.initial_balance
        self.current_balance = self.balance
        self.market_transaction_fee = 0.0005
        self.limit_transaction_fee = 0.0002
        self.position_open_steps = 0
        self.not_in_position_steps = 0
        self.collect_pnl = False

        # # Reset the iterator at the start of each episode to the beginning
        self.sequence_buffer.clear()
        self.load_initial_sequences()

        self.state = self.sequence_buffer[0]
        print("STATE RETURNED BY RESET METHOD : ", self.state.shape)

        return self.state, {}  # ensure returning a tuple with info

    def render(self, mode="human"):
        if mode == "human":
            # Set up the market colors and style for mplfinance
            mc = mpf.make_marketcolors(up="green", down="red", volume="gray")
            s = mpf.make_mpf_style(marketcolors=mc)

            # Define plot parameters
            fig, axes = plt.subplots()
            fig.subplots_adjust(bottom=0.2)

            def animate(i):
                # Clear the previous plot
                axes.clear()

                # Calculate start index for plotting to create a rolling window effect
                start_idx = max(0, self.current_step - self.input_length)
                end_idx = start_idx + self.input_length

                # Create a pandas DataFrame for mplfinance from the current state's slice
                columns = [
                    "Open",
                    "High",
                    "Low",
                    "Close",
                    "Volume",
                ]  # Update this list based on your state array's structure
                data_window = pd.DataFrame(
                    self.state[start_idx:end_idx, :4],
                    columns=columns,
                )  # Assuming the first 4 columns are OHLC
                data_window.index = pd.date_range(
                    start="now", periods=self.input_length, freq="T"
                )  # Generate a datetime index

                # Fetch current trade information if any
                trade_info = self.get_current_trade_info()

                # Add additional plots like moving averages or indicators if necessary
                # Note: This example just reuses existing data, adjust as necessary
                apds = [
                    mpf.make_addplot(
                        data_window["Close"].rolling(window=5).mean(), secondary_y=False
                    )
                ]

                if trade_info:
                    # Adding horizontal line for the entry price
                    apds.append(
                        mpf.make_addplot(
                            [trade_info["entry_price"]] * len(data_window),
                            type="line",
                            color="blue",
                            alpha=0.5,
                            width=2.0,
                        )
                    )

                # Plot the updated chart
                mpf.plot(
                    data_window,
                    type="candle",
                    style=s,
                    volume=True,
                    addplot=apds,
                    ax=axes,
                    title=f"Current Balance: ${self.balance:.2f}",
                    ylabel="Price ($)",
                )

                # Update current step for the next frame
                self.current_step = (self.current_step + 1) % (
                    len(self.state) - self.input_length
                )

            # Create an animation by repeatedly calling animate function
            ani = animation.FuncAnimation(
                fig, animate, frames=200, interval=100, repeat=True
            )
            plt.show()

    def get_current_trade_info(self):
        if self.position != 0:  # Check if there's an active position
            return {
                "entry_price": self.entry_price,
                "position_size": self.position_size,
                "leverage": self.leverage,
                "stop_loss": self.stop_loss,
                "take_profit": self.take_profit,
                "is_long": self.position > 0,  # True if long, False if short
            }
        return None  # No active trade
