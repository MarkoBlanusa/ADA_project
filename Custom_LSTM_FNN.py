import numpy as np
import pandas as pd
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch


torch, nn = try_import_torch()


class HybridModel(RecurrentNetwork, TorchModelV2, nn.Module):
    def __init__(
        self,
        obs_space,
        action_space,
        num_outputs,
        model_config,
        name,
        **customized_model_kwargs,
    ):
        # Initialize parent classes
        TorchModelV2.__init__(
            self, obs_space, action_space, num_outputs, model_config, name
        )
        nn.Module.__init__(self)

        # Retrieve configuration parameters or set default
        self.lstm_size = model_config.get("custom_model_config", {}).get(
            "lstm_size", 256
        )
        self.fnn_size = model_config.get("custom_model_config", {}).get("fnn_size", 128)

        self.seq_input_length = model_config.get("custom_model_config", {}).get(
            "seq_input_length", 96
        )

        # LSTM for time-series data
        self.lstm = nn.LSTM(
            input_size=self.seq_input_length,
            hidden_size=self.lstm_size,
            num_layers=1,
            batch_first=True,
        )

        # FNN for action-dependent states
        self.fnn = nn.Sequential(
            nn.Linear(7, self.fnn_size),  # 7 features for action-dependent state
            nn.ReLU(),
            nn.Linear(self.fnn_size, self.fnn_size),
        )

        # Define output layers for each action type
        self.action_heads = nn.ModuleDict(
            {
                "trade_action": nn.Linear(
                    self.lstm_size + self.fnn_size, action_space["trade_action"].n
                ),
                "position_proportion": nn.Linear(self.lstm_size + self.fnn_size, 1),
                "leverage_limit": nn.Linear(self.lstm_size + self.fnn_size, 1),
                "stop_loss_adj": nn.Linear(self.lstm_size + self.fnn_size, 1),
                "take_profit_adj": nn.Linear(self.lstm_size + self.fnn_size, 1),
                "stop_loss_pct": nn.Linear(self.lstm_size + self.fnn_size, 1),
                "take_profit_pct": nn.Linear(self.lstm_size + self.fnn_size, 1),
            }
        )

        # Fully connected layer for the value function
        self.value_head = nn.Linear(self.lstm_size + self.fnn_size, 1)

        self._value_out = None

    @override(RecurrentNetwork)
    def forward(self, input_dict, state, seq_lens):

        # Extract sequential and static inputs from the batch
        lstm_input = input_dict["obs"]["sequential"]  # Sequential features
        fnn_input = input_dict["obs"]["static"]  # Static features

        print("LSTM_INPUT SHAPE FIRST: ", lstm_input.shape)
        print("FNN_INPUT SHAPE FIRST: ", fnn_input.shape)

        print("LSTM_INPUT SHAPE: ", lstm_input.shape)
        print("FNN_INPUT SHAPE: ", fnn_input.shape)
        # Debugging sizes
        print("LSTM_INPUT SHAPE (before packing):", lstm_input.shape)
        # Additional debugging before packing
        print("Sequence lengths before packing:", seq_lens)
        print("Input size before packing:", lstm_input.size())

        if "prev_actions" in input_dict and "prev_rewards" in input_dict:

            print("PREV_ACTIONS SHAPE BEFORE: ", input_dict["prev_actions"].shape)
            print("PREV_REWARDS SHAPE BEFORE: ", input_dict["prev_rewards"].shape)

            # Ensure prev_actions and prev_rewards match the lstm_input's batch and sequence length
            batch_size, seq_length = lstm_input.shape[0], lstm_input.shape[1]
            action_features = input_dict["prev_actions"].shape[-1]

            prev_actions = (
                input_dict["prev_actions"]
                .unsqueeze(1)
                .expand(batch_size, seq_length, action_features)
            )
            prev_rewards = (
                input_dict["prev_rewards"]
                .unsqueeze(1)
                .unsqueeze(2)
                .expand(batch_size, seq_length, 1)
            )

            print("PREV_ACTIONS SHAPE AFTER RESHAPE: ", prev_actions.shape)
            print("PREV_REWARDS SHAPE AFTER RESHAPE: ", prev_rewards.shape)

            # Concatenate along the feature dimension (last dimension)
            lstm_input = torch.cat([lstm_input, prev_actions, prev_rewards], dim=-1)

        seq_lens = torch.full(
            (batch_size,), 96, dtype=torch.long
        )  # Fill with 96 for each sequence in the batch

        # Flatten parameters for efficiency if using CUDA
        if lstm_input.is_cuda:
            self.lstm.flatten_parameters()

        # Check and prepare hidden states if not already provided
        if not state:
            h0 = torch.zeros(
                self.lstm.num_layers, lstm_input.size(0), self.lstm.hidden_size
            ).to(lstm_input.device)
            c0 = torch.zeros(
                self.lstm.num_layers, lstm_input.size(0), self.lstm.hidden_size
            ).to(lstm_input.device)
        else:
            h0, c0 = state

        print("Sequence lengths reshaped before packing:", seq_lens)

        # Handling varying sequence lengths
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            lstm_input, seq_lens.cpu(), batch_first=True, enforce_sorted=False
        )

        print("Packed Input Size:", packed_input.data.size())
        print("Initial Hidden State Size:", h0.size())
        print("Initial Cell State Size:", c0.size())

        try:
            packed_output, (h, c) = self.lstm(packed_input, (h0, c0))
            lstm_out, _ = torch.nn.utils.rnn.pad_packed_sequence(
                packed_output, batch_first=True
            )
            print("LSTM_OUTPUT SHAPE (after unpacking):", lstm_out.shape)
        except RuntimeError as e:
            print(f"Error processing LSTM: {e}")
            raise

        lstm_out = lstm_out[:, -1, :]  # Select the output for the last timestep

        # Process with FNN
        fnn_out = self.fnn(fnn_input)

        # Combine and produce outputs
        combined_out = torch.cat([lstm_out, fnn_out], dim=1)

        # Additional print statements to track output sizes through layers
        print("LSTM_OUT SHAPE:", lstm_out.shape)
        print("FNN_OUT SHAPE:", fnn_out.shape)
        print("COMBINED_OUT SHAPE:", combined_out.shape)

        action_scores = {
            key: head(combined_out) for key, head in self.action_heads.items()
        }
        self._value_out = self.value_head(combined_out)

        return action_scores, [h.squeeze(0), c.squeeze(0)]

    @override(TorchModelV2)
    def value_function(self):
        assert self._value_out is not None, "Value function called before forward pass."
        return self._value_out.squeeze(-1)

    @override(TorchModelV2)
    def get_initial_state(self):
        # Return initial hidden and cell states for LSTM
        return [
            torch.zeros(1, self.lstm_size, dtype=torch.float32),
            torch.zeros(1, self.lstm_size, dtype=torch.float32),
        ]
