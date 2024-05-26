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
from ray import data
import os

logging.basicConfig(level=logging.DEBUG)


project_dir = r"C:\Users\marko\OneDrive\Bureau\Marko_documents\Etudes\Master_1ère\2ème_semestre\Advanced Data Analytics\Project"
hardcoded_dataset = np.load(os.path.join(project_dir, "normalized_test_data.npy"))


class TradingEnvironment(gymnasium.Env):

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        input_length=40,
        render_mode=None,
    ):
        super(TradingEnvironment, self).__init__()

        assert render_mode is None or render_mode in self.metadata["render_modes"]
        self.render_mode = render_mode

        # Trading states
        self.normalized_data = hardcoded_dataset
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
        self.balance = 10000
        self.current_balance = 10000
        self.previous_balance = 10000
        self.market_transaction_fee = 0.0005
        self.limit_transaction_fee = 0.0002
        self.position_open_steps = 0
        self.not_in_position_steps = 0
        self.number_account_blown = 0
        self.collect_pnl = False
        self.risk_reward_applied = False
        self.just_exited_trade = False
        self.trade_reward_applied = False
        self.is_exit = False

        self.trades = []
        self.balance_history = []
        self.entry_prices = [np.nan] * len(self.normalized_data)
        self.exit_prices = [np.nan] * len(self.normalized_data)
        self.entry_arrows = []
        self.exit_arrows = []

        self.trades = []
        self.balance_history = []

        self.trade_number = 0

        self.trade_info = {
            "trade_number": None,
            "entry_price": None,
            "exit_price": None,
            "position_size": None,
            "leverage": None,
            "stop_loss": None,
            "take_profit": None,
            "is_long": None,
            "pnl": None,
            "balance": None,
            "current_balance": None,
        }

        self.state = self.normalized_data[self.current_step]
        # Print the initial state for debugging
        print("INITIAL STATE FROM INIT :", self.state)
        # Calculate the total number of features by summing features across all timeframes
        self.num_features = self.state.shape[1]

        # Action space definition
        self.action_space = spaces.Dict(
            {
                "position_proportion": spaces.Box(
                    low=-1, high=1, shape=(1,), dtype=np.float32
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

        # Observation space definition
        max_val = 1.0
        min_val = 0.0
        self.observation_space = spaces.Box(
            low=min_val,
            high=max_val,
            shape=(input_length, self.num_features),
            dtype=np.float32,
        )

    def step(self, action):

        if self.current_step <= 10:
            print(f"ACTIONS FROM STEP NUMBER {self.current_step}: ", action)
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
        self.collect_pnl = False
        self.is_exit = False

        # Check first for an open position to close before checking for an entry in order to avoid an instant close after entering the position
        # Determine if stop loss or take profit was hit, or if the position has been liquidated
        if self.position != 0:

            self.position_open_steps += 1
            # print("TIME IN POSITION : ", self.position_open_steps)

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
            # print("CURRENT BALANCE : ", self.current_balance)

            if -unrealized_pnl >= 0.95 * self.balance:
                self.exit_price = (
                    worst_case_price  # Assuming immediate liquidation at current price
                )
                pnl = unrealized_pnl  # Final P&L after liquidation
                transaction_cost = (
                    self.market_transaction_fee * self.position_size * self.exit_price
                )
                self.balance += pnl - transaction_cost
                self.current_balance = self.balance
                self.entry_price = 0
                self.leverage = 0

                # Add an exit arrow
                self.exit_arrows.append(
                    (self.current_step, self.exit_price, self.position)
                )

                self.position = 0
                self.position_open_steps = 0
                self.position_size = 0
                self.stop_loss = 0
                self.take_profit = 0
                self.collect_pnl = True
                self.just_exited_trade = True
                self.is_exit = True

                # print("EXITING THE TRADE, EXIT PRICE : ", self.exit_price)
                # print("BALANCE ACCOUNT : ", self.balance)
                # print("TRANSACTION COSTS : ", -transaction_cost)

                # Log exit
                self.trade_info.update(
                    {
                        "exit_price": self.exit_price,
                        "pnl": pnl,
                        "balance": self.balance,
                        "current_balance": self.current_balance,
                    }
                )
                self.exit_prices[self.current_step] = self.exit_price

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
                transaction_cost = (
                    self.limit_transaction_fee * self.position_size * self.exit_price
                )
                self.balance += pnl
                self.balance -= transaction_cost
                self.current_balance = self.balance

                # Add an exit arrow
                self.exit_arrows.append(
                    (self.current_step, self.exit_price, self.position)
                )

                # print(
                #     "PERCENTAGE RISKED AT END: ",
                #     (
                #         (
                #             (
                #                 abs(self.stop_loss - self.entry_price)
                #                 * self.position_size
                #             )
                #             + transaction_cost
                #         )
                #         / self.balance
                #     )
                #     * 100,
                # )

                self.position = 0  # Reset position
                self.leverage = 0
                self.position_size = 0
                self.entry_price = 0
                self.stop_loss = 0
                self.take_profit = 0
                self.position_open_steps = 0  # Reset position duration counter
                self.collect_pnl = True
                self.just_exited_trade = True
                self.is_exit = True

                # print("EXITING THE TRADE, EXIT PRICE : ", self.exit_price)
                # print("BALANCE ACCOUNT : ", self.balance)
                # print("TRANSACTION COSTS : ", -transaction_cost)

                # Log exit
                self.trade_info.update(
                    {
                        "exit_price": self.exit_price,
                        "pnl": pnl,
                        "balance": self.balance,
                        "current_balance": self.current_balance,
                    }
                )
                self.exit_prices[self.current_step] = self.exit_price

            # Adjust stop loss and take profit based on current high/low if nothing made our position close
            if self.position > 0:
                self.stop_loss += stop_loss_adj * current_low
                self.take_profit += take_profit_adj * current_high
            elif self.position < 0:
                self.stop_loss -= stop_loss_adj * current_high
                self.take_profit -= take_profit_adj * current_low

            # print("STOP LOSS ADJUSTEMENT : ", self.stop_loss)
            # print("TAKE PROFIT ADJUSTEMENT : ", self.take_profit)

            # Log changes

            self.trade_info.update(
                {
                    "stop_loss": self.stop_loss,
                    "take_profit": self.take_profit,
                }
            )

        # Check for an entry
        if (
            not self.just_exited_trade
            and position_proportion != 0
            and self.position == 0
            and self.balance > 0
        ):
            self.position = 1 if position_proportion > 0 else -1
            self.entry_price = current_close * (1 + slippage)
            max_position_size = self.balance * self.leverage_limit / self.entry_price
            self.position_size = max_position_size * abs(position_proportion)
            self.stop_loss = self.entry_price * (
                1 - stop_loss_pct if self.position > 0 else 1 + stop_loss_pct
            )
            self.take_profit = self.entry_price * (
                1 + take_profit_pct if self.position > 0 else 1 - take_profit_pct
            )
            transaction_cost = (
                self.market_transaction_fee * self.position_size * self.entry_price
            )

            # Calculate the risk of the trade
            risk_per_trade = abs(self.entry_price - self.stop_loss) * self.position_size

            risk_percent = (risk_per_trade + transaction_cost) / self.balance

            # Check if the risk is at least 0.01% of the balance and minimum 1 USD to enter a position
            if risk_percent < 0.0001 or risk_per_trade < 1:
                self.position_size = 0
                self.position = 0  # Do not enter a trade if the risk is too low
                self.stop_loss = 0
                self.take_profit = 0
                transaction_cost = 0

            else:
                self.leverage = self.position_size * self.entry_price / self.balance
                self.balance -= transaction_cost
                self.current_balance = self.balance
                self.trade_number += 1
                self.not_in_position_steps = 0
                self.trade_reward_applied = True
                # print("OBERVATION NUMBER : ", self.current_step)
                # print("NEW ENTRY, TRADE NUMBER : ", self.trade_number)
                # print("ACTION TAKEN : ", trade_action)
                # print("ENTRY PRICE : ", self.entry_price)
                # print("POSITION SIZE : ", self.position_size)
                # print("STOP LOSS SET : ", self.stop_loss)
                # print("TAKE PROFIT SET : ", self.take_profit)
                # print("BALANCE ACCOUNT : ", self.balance)
                # print("TRANSACTION COSTS : ", -transaction_cost)
                # print(
                #     "PERCENTAGE RISKED : ",
                #     (
                #         (
                #             (
                #                 abs(self.stop_loss - self.entry_price)
                #                 * self.position_size
                #             )
                #             + transaction_cost
                #         )
                #         / self.balance
                #     )
                #     * 100,
                # )

                # Log entry
                self.trade_info.update(
                    {
                        "trade_number": self.trade_number,
                        "entry_price": self.entry_price,
                        "position_size": self.position_size,
                        "leverage": self.leverage,
                        "stop_loss": self.stop_loss,
                        "take_profit": self.take_profit,
                        "is_long": self.position > 0,
                        "exit_price": None,
                        "pnl": None,
                        "balance": self.balance,
                        "current_balance": self.current_balance,
                    }
                )
                self.entry_prices[self.current_step] = self.entry_price

                # Add an entry arrow
                self.entry_arrows.append(
                    (self.current_step, self.entry_price, self.position)
                )

        # When no positions are open and we aren't trading
        if self.position == 0:
            self.not_in_position_steps += 1
            if self.current_step <= 10:
                print("TIME NO POSITION : ", self.not_in_position_steps)

        self.just_exited_trade = False

        # Check for balance depletion to reset faster and penalize

        good_bad_reset_penalty = self.check_reset_conditions()

        if self.current_step <= 10:

            print("NUMBER ACCOUNT BLOWN : ", self.number_account_blown)

        # Make sure the balance stays under the bounds

        self.balance = min(max(self.balance, 0), 1e7)
        self.current_balance = min(max(self.current_balance, 0), 1e7)

        # Start collecting and updating everything related to the end of the step

        self.trades.append(self.trade_info)
        self.balance_history.append(self.balance)

        reward = self.calculate_reward(action, pnl)
        reward += good_bad_reset_penalty

        self.current_step += 1
        terminated = self.current_step >= (self.normalized_data.shape[0] - 1)

        if terminated:
            next_state = self.normalized_data[
                self.current_step - 1
            ]  # Use the last state before termination
        else:
            next_state = self.normalized_data[self.current_step]
        self.state = next_state
        info = {}

        truncated = False

        # # Print the updated state for debugging
        # print(f"Step {self.current_step}, New State: {self.state}")

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
            reset_penalty = -1000
            self.number_account_blown += 1
            return reset_penalty
        elif self.balance >= 1e7:
            self.balance = self.initial_balance
            reward_max_balance = 0
            return reward_max_balance
        else:
            return 0

    def get_aggregated_trade_info(self):
        print(self.trade_info)

    def calculate_reward(self, action, pnl):
        # Apply tanh to the basic reward to scale it between -1 and 1
        basic_reward = np.log(self.balance / self.previous_balance)
        # basic_reward = np.tanh(basic_reward)

        # # Penalty for illegal actions
        # illegal_penalty = 0

        # if self.position == 0 and (
        #     action["stop_loss_adj"] != 0 or action["take_profit_adj"] != 0
        # ):
        #     illegal_penalty = 0.01  # Maximum penalty for illegal actions

        # # Normalize time penalty between 0 and 1
        # time_penalty = self.position_open_steps / self.normalized_data.shape[0]

        # # Normalize inaction penalty, increasing over time
        # nothing_penalty = (
        #     self.not_in_position_steps / self.normalized_data.shape[0]
        # ) ** 0.5  # Exponential growth

        # # Reward for taking action
        # action_reward = (
        #     0.001 if action["position_proportion"] != 0 else -0.001
        # )  # Reward for action, penalty for inaction

        # # Reward for the number of trades
        # nb_trades_reward = 0.01 if self.trade_reward_applied else 0
        # if self.trade_reward_applied:
        #     self.trade_reward_applied = False

        # # Determine acceptable risk percentage based on balance
        # def acceptable_risk(balance):
        #     if balance <= 10000:
        #         return 0.02
        #     elif balance <= 100000:
        #         return 0.01
        #     elif balance <= 1000000:
        #         return 0.005
        #     else:
        #         return 0.0025

        # # Normalize risk penalty between 0 and 1
        # max_risk_percent = acceptable_risk(self.balance)
        # if self.position > 0:
        #     risk_per_trade = (
        #         (self.entry_price - self.stop_loss) * self.position_size
        #     ) / self.balance
        # elif self.position < 0:
        #     risk_per_trade = (
        #         (self.stop_loss - self.entry_price) * self.position_size
        #     ) / self.balance
        # else:
        #     risk_per_trade = 0

        # risk_penalty = max(0, risk_per_trade - max_risk_percent) * 0.1
        # risk_penalty = np.tanh(risk_penalty)  # Scale risk penalty using tanh

        # # Normalize trade efficiency reward
        # trade_efficiency_reward = 0
        # if self.risk_reward_applied == False or self.is_exit:
        #     if risk_per_trade > 0:
        #         trade_efficiency = pnl / (risk_per_trade * self.balance)
        #     else:
        #         trade_efficiency = 0

        #     trade_efficiency_reward = max(0, trade_efficiency - 1) * 0.01
        #     trade_efficiency_reward = np.tanh(
        #         trade_efficiency_reward
        #     )  # Scale trade efficiency using tanh

        #     if self.is_exit:
        #         risk_penalty *= 2  # Double the risk penalty on exit
        #         trade_efficiency_reward *= (
        #             2  # Double the trade efficiency reward on exit
        #         )

        #     self.risk_reward_applied = True

        # if self.position == 0:
        #     self.risk_reward_applied = False

        # # Combine all components to form the final reward
        return basic_reward
        #     - time_penalty
        #     - nothing_penalty
        #     - illegal_penalty
        #     - risk_penalty
        #     + trade_efficiency_reward
        #     + nb_trades_reward
        #     + action_reward
        # )

    def reset(self, seed=None, **kwargs):

        super().reset(seed=seed, **kwargs)  # Call to super to handle seeding properly

        # logging.debug(f"Current iteration: {self.iteration_count}")

        self.trades = []
        self.balance_history = []
        self.entry_prices = [np.nan] * len(self.normalized_data)
        self.exit_prices = [np.nan] * len(self.normalized_data)
        self.entry_arrows = []
        self.exit_arrows = []

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
        self.balance = 10000
        self.current_balance = 10000
        self.previous_balance = 10000
        self.market_transaction_fee = 0.0005
        self.limit_transaction_fee = 0.0002
        self.position_open_steps = 0
        self.not_in_position_steps = 0
        self.collect_pnl = False
        self.risk_reward_applied = False
        self.just_exited_trade = False
        self.trade_reward_applied = False

        self.trade_number = 0

        self.trade_info = {
            "trade_number": None,
            "entry_price": None,
            "exit_price": None,
            "position_size": None,
            "leverage": None,
            "stop_loss": None,
            "take_profit": None,
            "is_long": None,
            "pnl": None,
            "balance": None,
            "current_balance": None,
        }

        # # Reset the iterator at the start of each episode to the beginning
        # self.data_iterator = iter(
        #     self.normalized_data.iter_batches(
        #         batch_size=128, prefetch_batches=self.prefetch_batches
        #     )
        # )
        # self.state = next(self.data_iterator)["data"][0]  # Re-fetch the first sequence
        # self.iterator = iter(self.normalized_data)
        # self.state = next(self.iterator).squeeze(0)
        self.state = self.normalized_data[self.current_step]

        print("INITIAL STATE FROM RESET : ", self.state)

        return self.state, {}  # ensure returning a tuple with info

    # # Function to dynamically adjust prefetch_batches
    # def adjust_prefetch(self, dataset, batch_size, memory_fraction=0.5):
    #     # Retrieve Ray's allocated memory (assuming Ray has been initialized)
    #     ray_memory = 6 * (1024**3)

    #     # Estimate memory usage per batch (assuming some average size per observation)
    #     memory_per_observation = dataset.size_bytes() / len(dataset)
    #     estimated_batch_memory = memory_per_observation * batch_size

    #     # Calculate optimal prefetch_batches
    #     optimal_prefetch = int((ray_memory * memory_fraction) / estimated_batch_memory)

    #     if optimal_prefetch != self.prefetch_batches:
    #         self.prefetch_batches = optimal_prefetch

    def render(self, mode="human"):

        # Create a DataFrame for OHLC data
        ohlc_data = []
        for i in range(len(self.normalized_data)):
            open_price = self.normalized_data[i][-1][0]
            high_price = self.normalized_data[i][-1][1]
            low_price = self.normalized_data[i][-1][2]
            close_price = self.normalized_data[i][-1][3]
            ohlc_data.append([i, open_price, high_price, low_price, close_price])

        ohlc_df = pd.DataFrame(
            ohlc_data, columns=["Date", "Open", "High", "Low", "Close"]
        )
        ohlc_df["Date"] = pd.to_datetime(ohlc_df["Date"], unit="s")
        ohlc_df.set_index("Date", inplace=True)

        # Create a figure for the plot
        fig, ax = plt.subplots(figsize=(14, 7))

        # Plot OHLC data
        mpf.plot(ohlc_df, type="candle", ax=ax, style="charles", show_nontrading=True)

        # Plot entry and exit points
        marker_offset = 0.3  # Adjust this value as needed

        for entry in self.entry_arrows:
            step, price, position = entry
            color = "green" if position > 0 else "red"
            marker_price = ohlc_df.iloc[step]["High"] * (
                1 + marker_offset
            )  # Place marker above the high price
            ax.scatter(
                ohlc_df.index[step],
                marker_price,
                color=color,
                marker="v" if position > 0 else "v",
                label="Long Entry" if position > 0 else "Short Entry",
            )

        for exit in self.exit_arrows:
            step, price, position = exit
            color = "red" if position > 0 else "green"
            marker_price = ohlc_df.iloc[step]["Low"] * (
                1 - marker_offset
            )  # Place marker below the low price
            ax.scatter(
                ohlc_df.index[step],
                marker_price,
                color=color,
                marker="^" if position > 0 else "^",
                label="Long Exit" if position > 0 else "Short Exit",
            )

        # Ensure labels are unique in the legend
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))
        ax.legend(by_label.values(), by_label.keys())

        ax.set_title("OHLC Candlestick Chart with Trade Markers")
        ax.set_xlabel("Time")
        ax.set_ylabel("Price")
        ax.grid(True)

        plt.tight_layout()
        plt.show()

        # Create a separate figure for the balance history
        plt.figure(figsize=(10, 5))
        plt.plot(self.balance_history, label="Balance", color="blue")
        plt.title("Evolution of Current Balance")
        plt.xlabel("Time Steps")
        plt.ylabel("Balance")
        plt.legend()
        plt.grid(True)
        plt.show()

    # def init_plot(self):
    #     # Initial drawing of the plot, to be called once at the beginning of the animation
    #     self.axes.clear()
    #     self.plot_data()
    #     return self.axes

    # def animate(self, i):
    #     self.current_step = i
    #     self.plot_data()

    # def plot_data(self):
    #     # Clear the previous plot
    #     self.axes.clear()

    #     # Calculate start index for plotting to create a rolling window effect
    #     start_idx = max(0, self.current_step - self.input_length)
    #     end_idx = start_idx + self.input_length

    #     # Create a pandas DataFrame for mplfinance from the current state's slice
    #     columns = ["Open", "High", "Low", "Close", "Volume"]
    #     data_window = pd.DataFrame(
    #         self.state[start_idx:end_idx, :5], columns=columns
    #     )  # Assuming the first 4 columns are OHLC
    #     data_window.index = pd.date_range(
    #         start="now", periods=self.input_length, freq="T"
    #     )  # Generate a datetime index

    #     # Fetch current trade information if any
    #     trade_info = self.get_current_trade_info()

    #     # Add additional plots like moving averages or indicators if necessary
    #     apds = [
    #         mpf.make_addplot(
    #             data_window["Close"].rolling(window=5).mean(), secondary_y=False
    #         )
    #     ]

    #     if trade_info:
    #         # Adding horizontal line for the entry price
    #         apds.append(
    #             mpf.make_addplot(
    #                 [trade_info["entry_price"]] * len(data_window),
    #                 type="line",
    #                 color="blue",
    #                 alpha=0.5,
    #                 width=2.0,
    #                 ax=self.axes,
    #             )
    #         )

    #     # Plot the updated chart
    #     mpf.plot(
    #         data_window,
    #         type="candle",
    #         style=self.s,
    #         volume=True,
    #         addplot=apds,
    #         ax=self.axes,
    #         title=f"Current Balance: ${self.current_balance:.2f} | Trades: {self.trade_number}",
    #         ylabel="Price ($)",
    #     )

    #     # Update current step for the next frame
    #     self.current_step = (self.current_step + 1) % (
    #         len(self.state) - self.input_length
    #     )
    #     return self.axes

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
