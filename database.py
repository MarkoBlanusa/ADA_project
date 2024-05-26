from typing import *
import logging
import time

import h5py
import numpy as np
import pandas as pd


logger = logging.getLogger()

logging.basicConfig()
log = logging.getLogger(__name__)
log.setLevel(logging.INFO)
log.info("%s logger started.", __name__)

TF_EQUIV = {
    "1m": "1Min",
    "5m": "5Min",
    "15m": "15Min",
    "30m": "30Min",
    "1h": "1H",
    "4h": "4H",
    "12h": "12H",
    "1d": "D",
}


def resample_timeframe(data: pd.DataFrame, tf: str) -> pd.DataFrame:
    return data.resample(TF_EQUIV[tf]).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )


class Hdf5client:
    def __init__(self, exchange: str):
        self.hf = h5py.File(f"data/{exchange}.h5", "a")
        self.hf.flush()

    def create_dataset(self, symbol: str):
        if symbol not in self.hf.keys():
            self.hf.create_dataset(symbol, (0, 6), maxshape=(None, 6), dtype="float64")
            self.hf.flush()

    def write_data(self, symbol: str, data: List[Tuple]):

        min_ts, max_ts = self.get_first_last_timestamp(symbol)

        if min_ts is None:
            min_ts = float("inf")
            max_ts = 0

        filtered_data = []

        for d in data:
            if d[0] < min_ts:
                filtered_data.append(d)
            elif d[0] > max_ts:
                filtered_data.append(d)

        if len(filtered_data) == 0:
            logger.warning("%s: No data to insert", symbol)

        data_array = np.array(data)

        self.hf[symbol].resize(self.hf[symbol].shape[0] + data_array.shape[0], axis=0)
        self.hf[symbol][-data_array.shape[0] :] = data_array

        self.hf.flush()

    def get_data(
        self, symbol: str, from_time: int, to_time: int
    ) -> Union[None, pd.DataFrame]:

        start_query = time.time()

        existing_data = self.hf[symbol][:]

        if len(existing_data) == 0:
            return None

        data = sorted(existing_data, key=lambda x: x[0])
        data = np.array(data)

        df = pd.DataFrame(
            data, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df = df[(df["timestamp"] >= from_time) & (df["timestamp"] <= to_time)]

        df["timestamp"] = pd.to_datetime(
            df["timestamp"].values.astype(np.int64), unit="ms"
        )
        df.set_index("timestamp", drop=True, inplace=True)

        query_time = round((time.time() - start_query), 2)

        logger.info(
            "Retrieved %s %s data from database in %s seconds",
            len(df.index),
            symbol,
            query_time,
        )

        return df

    def get_data2(
        self, symbol: str, from_time: int, to_time: int, tf: str
    ) -> Union[None, pd.DataFrame]:

        start_query = time.time()

        if tf == "1m":
            params = 60
        elif tf == "5m":
            params = 5 * 60
        elif tf == "15m":
            params = 15 * 60
        elif tf == "30m":
            params = 30 * 60
        elif tf == "1h":
            params = 60 * 60
        elif tf == "4h":
            params = 4 * 60 * 60
        elif tf == "12h":
            params = 12 * 60 * 60
        elif tf == "1d":
            params = 24 * 60 * 60

        existing_data = self.hf[symbol][:]

        if len(existing_data) == 0:
            return None

        data = sorted(existing_data, key=lambda x: x[0])
        data = np.array(data)

        df = pd.DataFrame(
            data, columns=["timestamp", "open", "high", "low", "close", "volume"]
        )
        df = df[
            (df["timestamp"] >= (from_time - params * 1000))
            & (df["timestamp"] <= (to_time - params * 1000))
        ]

        df["timestamp"] = pd.to_datetime(
            df["timestamp"].values.astype(np.int64), unit="ms"
        )
        df.set_index("timestamp", drop=True, inplace=True)

        df = resample_timeframe(df, tf)

        query_time = round((time.time() - start_query), 2)

        logger.info(
            "Retrieved %s %s data from database in %s seconds",
            len(df.index),
            symbol,
            query_time,
        )

        return df

    def get_first_last_timestamp(
        self, symbol: str
    ) -> Union[Tuple[None, None], Tuple[float, float]]:

        existing_data = self.hf[symbol][:]

        if len(existing_data) == 0:
            return None, None

        first_ts = min(existing_data, key=lambda x: x[0])[0]
        last_ts = max(existing_data, key=lambda x: x[0])[0]

        return first_ts, last_ts
