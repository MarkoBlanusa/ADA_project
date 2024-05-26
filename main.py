import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import datetime

from binance import BinanceClient
from database import Hdf5client
import data_collector
import utils
from trade_env_numpy import TradingEnvironment

import ray
from ray import tune
from ray.rllib.models import ModelCatalog
from ray.rllib.algorithms.ppo import PPOConfig
from ray.rllib.models.torch.torch_modelv2 import TorchModelV2
from ray.tune.registry import register_env
from ray.rllib.models.torch.recurrent_net import RecurrentNetwork
from ray.rllib.utils.annotations import override
from ray.rllib.utils.framework import try_import_torch
from ray.tune.schedulers import HyperBandScheduler
from ray.tune.search import basic_variant
from ray.rllib.algorithms.ppo import PPO


from LSTMclass import LSTMModel

from gymnasium.envs.registration import register

# initialize torch and neural networks
torch, nn = try_import_torch()


# Data retrieving
def get_timeframe_data(symbol, from_time, to_time, timeframe):
    h5_db = Hdf5client("binance")
    data = h5_db.get_data(symbol, from_time, to_time)
    if timeframe != "1m":
        data = utils.resample_timeframe(data, timeframe)
    return data


# Features engineering
def calculate_indicators(df):
    df["MA5"] = df["close"].rolling(window=5).mean()
    df["MA10"] = df["close"].rolling(window=10).mean()

    delta = df["close"].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    rs = gain / loss
    df["RSI"] = 100 - (100 / (1 + rs))

    exp1 = df["close"].ewm(span=12, adjust=False).mean()
    exp2 = df["close"].ewm(span=26, adjust=False).mean()
    df["MACD"] = exp1 - exp2

    df["BB_up"] = df["MA10"] + 2 * df["close"].rolling(window=10).std()
    df["BB_low"] = df["MA10"] - 2 * df["close"].rolling(window=10).std()

    df["high-low"] = df["high"] - df["low"]
    df["high-close"] = np.abs(df["high"] - df["close"].shift())
    df["low-close"] = np.abs(df["low"] - df["close"].shift())
    df["TR"] = df[["high-low", "high-close", "low-close"]].max(axis=1)
    df["ATR"] = df["TR"].rolling(window=14).mean()

    df.drop(["high-low", "high-close", "low-close", "TR"], axis=1, inplace=True)

    return df.dropna()


# Normalize the dataframes
def normalize_dataframes(
    data,
    ohlc_columns=["open", "high", "low", "close"],
    volume_column="volume",
    indicator_columns=["MA5", "MA10", "RSI", "MACD", "BB_up", "BB_low", "ATR"],
    epsilon=0.0001,  # Small constant to avoid zero in normalized data
):
    """
    Normalize the features of financial dataframes.

    :param data: A dictionary of pandas dataframes keyed by timeframe.
    :param ohlc_columns: List of columns to be normalized across all dataframes together.
    :param volume_column: The volume column to be normalized independently for each dataframe.
    :param indicator_columns: List of other indicator columns to normalize independently for each dataframe.
    :param epsilon: Small constant to set the lower bound of the normalized range.
    :return: The dictionary of normalized dataframes and the OHLC scaler used.
    """
    # Initialize the scalers
    ohlc_scaler = MinMaxScaler(
        feature_range=(epsilon, 1)
    )  # Set feature range with epsilon
    volume_scaler = MinMaxScaler(feature_range=(epsilon, 1))

    # Create a new dictionary to store the normalized dataframes
    normalized_data = {}

    # Normalize OHLC data across all timeframes together
    combined_ohlc = pd.concat([df[ohlc_columns] for df in data.values()], axis=0)
    scaled_ohlc = ohlc_scaler.fit_transform(combined_ohlc).astype(np.float32)

    # Distribute the normalized OHLC values back to the original dataframes
    start_idx = 0
    for tf, df in data.items():
        end_idx = start_idx + len(df)
        # Create a copy of the original dataframe to avoid modifying it
        normalized_df = df.copy()
        normalized_df[ohlc_columns] = scaled_ohlc[start_idx:end_idx]
        # Store the normalized dataframe in the new dictionary
        normalized_data[tf] = normalized_df
        start_idx = end_idx

    # Normalize volume independently for each timeframe
    for tf, df in normalized_data.items():
        volume_scaler = MinMaxScaler(
            feature_range=(epsilon, 1)
        )  # Reinitialize scaler for each dataframe
        df[volume_column] = volume_scaler.fit_transform(df[[volume_column]])

    # Normalize other indicators independently for each indicator within each timeframe
    for tf, df in normalized_data.items():
        for col in indicator_columns:
            if col in df.columns:
                scaler = MinMaxScaler(feature_range=(epsilon, 1))
                df[[col]] = scaler.fit_transform(df[[col]])

    return normalized_data, ohlc_scaler


# Add identifiers for the timeframes in order to help the LSTM to make the difference
# def add_timeframe_identifier(data_dict):
#    timeframe_ids = {
#        "15m": 0,
#        "30m": 1,
#        "1h": 2,
#        "1d": 3,
#    }
#    for timeframe, data in data_dict.items():
#        # Assuming `data` is a DataFrame
#        data["timeframe_id"] = timeframe_ids[timeframe]
#    return data_dict


def resample_to_frequency(df, freq):
    # Resample the dataframe to the specified frequency using forward-fill to handle NaNs
    return df.resample(freq).ffill()


# Create sequences and split them for the LSTM
def create_and_split_sequences(
    data_dict, input_length, validation_pct, test_pct, base_freq="D"
):
    # Resample all timeframes to the base frequency of 15 minutes
    resampled_data = {
        tf: resample_to_frequency(df, base_freq) for tf, df in data_dict.items()
    }

    # Align lengths by truncating to the shortest length after resampling
    min_length = min(len(df) for df in resampled_data.values())
    aligned_data = {
        tf: df.iloc[:min_length].reset_index(drop=True)
        for tf, df in resampled_data.items()
    }

    # Concatenate data from all timeframes
    concatenated_data = pd.concat(aligned_data.values(), axis=1)

    # Create sequences
    num_sequences = len(concatenated_data) - input_length + 1
    X = np.zeros(
        (num_sequences, input_length, concatenated_data.shape[1]), dtype=np.float32
    )

    for i in range(num_sequences):
        X[i] = concatenated_data.iloc[i : (i + input_length)].values

    # Split the data
    n = X.shape[0]
    test_index = int(n * (1 - test_pct))
    validation_index = int(n * (1 - test_pct - validation_pct))

    train_X = X[:validation_index]
    val_X = X[validation_index:test_index]
    test_X = X[test_index:]

    return (
        train_X.astype(np.float32),
        val_X.astype(np.float32),
        test_X.astype(np.float32),
    )


# Split the raw prices dataframe for the environment step logic
def split_time_series_data_dict(data_dict, validation_pct, test_pct):
    split_data = {}
    for timeframe, data in data_dict.items():
        # Ensure all columns are float32
        data = data.astype("float32")

        # Compute indices for splitting
        num_samples = len(data)
        test_index = int(num_samples * (1 - test_pct))
        validation_index = int(num_samples * (1 - test_pct - validation_pct))

        # Split the data
        train_data = data.iloc[:validation_index]
        validation_data = data.iloc[validation_index:test_index]
        test_data = data.iloc[test_index:]

        # Store the split data in the dictionary
        split_data[timeframe] = {
            "train": train_data,
            "validation": validation_data,
            "test": test_data,
        }

    return split_data


# Creates the environment for the ray library
def env_creator(env_config):
    return TradingEnvironment(
        # data=env_config["data"],
        normalized_data=env_config["normalized_data"],
        # ohlc_scaler=env_config["ohlc_scaler"],
        # timeframe=env_config.get("timeframe", "15m"),
        # mode=env_config.get("mode", "train"),
        input_length=env_config.get("input_length", 40),
    )


if __name__ == "__main__":

    from_time = "2019-10-04"
    to_time = "2024-04-13"
    symbol = "BTCUSDT"

    # Convert times
    from_time = int(
        datetime.datetime.strptime(from_time, "%Y-%m-%d").timestamp() * 1000
    )
    to_time = int(datetime.datetime.strptime(to_time, "%Y-%m-%d").timestamp() * 1000)

    # Define timeframes
    timeframes = ["1d"]
    dataframes = {}

    for tf in timeframes:
        dataframes[tf] = get_timeframe_data(symbol, from_time, to_time, tf)

    # Syncronize the timeframes after computing the features
    for tf in timeframes:
        dataframes[tf] = calculate_indicators(dataframes[tf]).dropna()

    latest_start_date = None
    earliest_end_date = None

    for df in dataframes.values():
        start_date = df.index.min()
        end_date = df.index.max()
        if latest_start_date is None or start_date > latest_start_date:
            latest_start_date = start_date
        if earliest_end_date is None or end_date < earliest_end_date:
            earliest_end_date = end_date

    # Ensure all DataFrames start and end on these dates
    for tf in dataframes:
        dataframes[tf] = dataframes[tf][
            (dataframes[tf].index >= latest_start_date)
            & (dataframes[tf].index <= earliest_end_date)
        ]

    # Normalize the dataframes and add identifiers in timeframes for the LSTM
    normalized_dataframes, ohlc_scaler = normalize_dataframes(dataframes)
    # normalized_dataframes = add_timeframe_identifier(normalized_dataframes)

    # Sequence and split the normalized data for the LSTM
    input_length = 40  # Define the length of the input window
    validation_pct = 0  # 0% validation set
    test_pct = 0.1  # 10% test set

    train_X, val_X, test_X = create_and_split_sequences(
        normalized_dataframes, input_length, validation_pct, test_pct
    )

    print("NUM OBSERVATIONS : ", len(train_X))

    # Split the raw prices dataset for the environment step method logic
    splitted_raw_prices = split_time_series_data_dict(
        dataframes, validation_pct=0, test_pct=0.1
    )

    # Register the environment in gymnasium
    register(
        id="trading_env:numpy-v0",
        entry_point="trade_env_numpy:TradingEnvironment",
    )

    # Register the environement in ray
    # register_env("trading_env2-v0", lambda config: env_creator(config))

    # env = TradingEnvironment(
    #    data=splitted_raw_prices, normalized_data=train_X, ohlc_scaler=ohlc_scaler
    # )

    # env.reset()

    ## Register the LSTM class in ray
    # ModelCatalog.register_custom_model("lstm_model", LSTMModel)

    # Start the PPO configuration and training

    # Define the environment creator function
    def env_creator(env_config):
        return TradingEnvironment(**env_config)

    # Register the custom environment
    register_env("trading_env_numpy-v0", env_creator)

    # Ensure Ray is properly initialized
    if ray.is_initialized():
        ray.shutdown()
    ray.init(ignore_reinit_error=True, object_store_memory=3 * (1024**3))

    # Define the search space
    # search_space = {
    #    "lr": tune.loguniform(1e-4, 1e-1),  # Learning rate
    #    "train_batch_size": tune.choice([1024, 2048]),
    #    "sgd_minibatch_size": tune.choice([64, 128]),
    #    "num_sgd_iter": tune.choice([10, 20]),
    #    "num_rollout_workers": tune.choice([1, 2]),
    #    "model": {
    #        "lstm_cell_size": tune.choice([64, 128]),
    #        "fcnet_hiddens": tune.choice([[64, 64], [128, 128]]),
    #    },
    # }

    # Scheduler to prune less promising trials
    # scheduler = HyperBandScheduler(
    #    time_attr="training_iteration",
    #    max_t=100,  # maximum iterations per configuration
    #    reduction_factor=3,
    #    stop_last_trials=True,
    # )

    # Configuration using PPOConfig
    config = PPOConfig()
    config.environment(
        env="trading_env_numpy-v0",
        env_config={
            # "data": splitted_raw_prices,
            "normalized_data": train_X,
            # "ohlc_scaler": ohlc_scaler,
            # "timeframe": "15m",
            # "mode": "train",
            "input_length": 40,
        },
    )
    config.framework("torch")
    config.resources(num_gpus=1, num_cpus_per_worker=3)
    config.rollouts(
        num_rollout_workers=4,
        rollout_fragment_length=360,
        batch_mode="complete_episodes",
    )
    config.training(
        gamma=0.99,
        lr=5e-2,
        train_batch_size=1440,
        sgd_minibatch_size=200,
        num_sgd_iter=30,
        shuffle_sequences=False,
        grad_clip=0.5,
        lambda_=0.99,
        entropy_coeff=0.003,
        clip_param=0.2,
    )
    # Access the model configuration directly via the `.model` attribute
    config.model["use_lstm"] = True
    config.model["lstm_cell_size"] = 32
    config.model["fcnet_hiddens"] = [16]
    config.model["fcnet_activation"] = "relu"
    config.model["post_fcnet_activation"] = "linear"
    config.model["lstm_use_prev_action_reward"] = True
    config.model["max_seq_len"] = 40
    config.model["_disable_action_flattening"] = True

    # config.model["shuffle_sequences"] = False  # Disable shuffling

    # Verify configuration
    # print(config.to_dict())  # This will print the current configuration as a dictionary

    # Run hyperparameter tuning with Ray Tune
    results = tune.run(
        "PPO",
        config=config,
        metric="episode_reward_mean",
        mode="max",
        stop={"training_iteration": 10000},
        # num_samples=1,  # Number of different sets of hyperparameters to try
        search_alg=basic_variant.BasicVariantGenerator(),  # Simple random search
        # scheduler=scheduler,
        verbose=1,
        checkpoint_freq=10,  # Save a checkpoint every 10 training iterations
        checkpoint_at_end=True,  # Ensure a checkpoint is saved at the end of training
        local_dir=r"C:\Users\marko\OneDrive\Bureau\Marko_documents\Etudes\Master_1ère\2ème_semestre\Advanced Data Analytics\Project\RL_checkpoints",
    )

    # Access the best trial's results and checkpoints
    best_trial = results.get_best_trial("episode_reward_mean", "max", "last")
    print("Best trial config: {}".format(best_trial.config))
    print(
        "Best trial final reward: {}".format(
            best_trial.last_result["episode_reward_mean"]
        )
    )
