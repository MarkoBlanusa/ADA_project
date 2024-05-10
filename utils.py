import datetime
from ctypes import *

import pandas as pd

TF_EQUIV = {"1m": "1Min", "5m": "5Min", "15m": "15Min", "30m": "30Min", "1h": "1H", "4h": "4H", "12h": "12H", "1d": "D"}

STRAT_PARAMS = {
    "obv": {
        "ma_period": {"name": "MA Period", "type": int, "min": 2, "max": 200},
    },
    "ichimoku": {
        "kijun": {"name": "Kijun Period", "type": int, "min": 25, "max": 90},
        "tenkan": {"name": "Tenkan Period", "type": int, "min": 10, "max": 50},
    },
    "sup_res": {
        "min_points": {"name": "Min. Points", "type": int, "min": 2, "max": 20},
        "min_diff_points": {"name": "Min. Difference between Points", "type": int, "min": 1, "max": 100},
        "rounding_nb": {"name": "Rounding Number", "type": float, "min": 0.001, "max": 0.05, "decimals": 3},
        "ratio": {"name": "WIN/LOSS Ratio", "type": float, "min": 0.5, "max": 10, "decimals": 2},
        "risk": {"name": "Risk per trade", "type": float, "min": 0.005, "max": 0.02, "decimals": 2},
        "leverage": {"name": "Leverage", "type": int, "min": 1, "max": 50},
        "cash": {"name": "Initial balance (USD)", "type": int, "min": 1000, "max": 1000},
        "stop_short": {"name": "Stop short safety", "type": float, "min": 1.001, "max": 1.01, "decimals": 4},
        "stop_long": {"name": "Stop long safety", "type": float, "min": 0.99, "max": 0.999, "decimals": 4}

    },
    "fractals": {
        "ema_fast": {"name": "EMA Fast", "type": int, "min": 6, "max": 50},
        "ema_middle": {"name": "EMA Middle", "type": int, "min": 40, "max": 100},
        "ema_slow": {"name": "EMA Slow", "type": int, "min": 100, "max": 400},
        "rsi_length": {"name": "RSI", "type": int, "min": 14, "max": 14},
        "stop_ratio": {"name": "WIN/LOSS Ratio", "type": float, "min": 1, "max": 3, "decimals": 2},
        "risk": {"name": "Risk per trade", "type": float, "min": 0.005, "max": 0.02, "decimals": 3},
        "cash": {"name": "Initial balance (USD)", "type": int, "min": 1000, "max": 1000},
        "leverage": {"name": "Leverage", "type": int, "min": 1, "max": 50},
    },
    "fractals2": {
        "ema_fast": {"name": "EMA Fast", "type": int, "min": 1, "max": 50},
        "ema_middle": {"name": "EMA Middle", "type": int, "min": 40, "max": 100},
        "ema_slow": {"name": "EMA Slow", "type": int, "min": 100, "max": 400},
        "rsi_length": {"name": "RSI", "type": int, "min": 1, "max": 100},
        "stop_ratio": {"name": "WIN/LOSS Ratio", "type": float, "min": 0.5, "max": 10, "decimals": 2},
        "risk": {"name": "Risk per trade", "type": float, "min": 0.005, "max": 0.02, "decimals": 3},
        "cash": {"name": "Initial balance (USD)", "type": int, "min": 1000, "max": 1000},
        "leverage": {"name": "Leverage", "type": int, "min": 1, "max": 125},
        "breakeven": {"name": "Break Even", "type": float, "min": 0.1, "max": 1, "decimals": 2}
    },
    "fractal_simple": {
        "ema_fast": {"name": "EMA Fast", "type": int, "min": 1, "max": 50},
        "ema_middle": {"name": "EMA Middle", "type": int, "min": 40, "max": 100},
        "ema_slow": {"name": "EMA Slow", "type": int, "min": 100, "max": 400},
        "rsi_period": {"name": "RSI", "type": int, "min": 14, "max": 14},
        "bk_ratio": {"name": "Break Even", "type": float, "min": 0.1, "max": 0.9, "decimals": 2},
        "ratio": {"name": "WIN/LOSS Ratio", "type": float, "min": 0.5, "max": 3, "decimals": 2},
        "risk": {"name": "Risk per trade", "type": float, "min": 0.005, "max": 0.02, "decimals": 3},
        "cash": {"name": "Initial balance (USD)", "type": int, "min": 1000, "max": 1000},
        "leverage": {"name": "Leverage", "type": int, "min": 1, "max": 125},
    },
    "sma": {
            "slow_ma": {"name": "Slow MA Period", "type": int, "min": 2, "max": 200},
            "fast_ma": {"name": "Fast MA Period", "type": int, "min": 2, "max": 200},
    },
    "psar": {
            "initial_acc": {"name": "Initial Acceleration", "type": float, "min": 0.005, "max": 0.2, "decimals": 2},
            "acc_increment": {"name": "Acceleration Increment", "type": float, "min": 0.005, "max": 0.2, "decimals": 2},
            "max_acc": {"name": "Max. Acceleration", "type": float, "min": 0.05, "max": 2, "decimals": 2},
    },
    "fractal_test": {
            "ema_fast": {"name": "EMA Fast", "type": int, "min": 2, "max": 8},
            "ema_middle": {"name": "EMA Middle", "type": int, "min": 90, "max": 110},
            "ema_slow": {"name": "EMA Slow", "type": int, "min": 100, "max": 200},
            "rsi_period": {"name": "RSI Period", "type": int, "min": 50, "max": 70},
            "rsi_long_value": {"name": "RSI Long Value Condition", "type": int, "min": 70, "max": 90},
            "rsi_short_value": {"name": "RSI Short Value Condition", "type": int, "min": 1, "max": 10},
            "ratio": {"name": "WIN/LOSS Ratio", "type": float, "min": 7, "max": 10, "decimals": 2},
            "risk": {"name": "Risk per trade", "type": float, "min": 0.02, "max": 0.02, "decimals": 3},
            "cash": {"name": "Initial balance (USD)", "type": int, "min": 1000, "max": 1000},
            "leverage": {"name": "Leverage", "type": int, "min": 1, "max": 15},
            "bk_ratio": {"name": "Breakeven ratio", "type": float, "min": 0.9, "max": 1, "decimals": 2},
            "dist_long": {"name": "Stop loss long distance", "type": float, "min": 0.999, "max": 1, "decimals": 4},
            "dist_short": {"name": "Stop loss short distance", "type": float, "min": 1, "max": 1.001, "decimals": 4},
            "nb_lows": {"name": "Down Trends precisions", "type": int, "min": 1, "max": 5},
            "nb_highs": {"name": "Up Trends precisions", "type": int, "min": 1, "max": 5},
            "macd_line_fast": {"name": "MACD Fast line", "type": int, "min": 12, "max": 12},
            "macd_line_slow": {"name": "MACD Slow line", "type": int, "min": 26, "max": 26},
            "macd_signal": {"name": "MACD Signal", "type": int, "min": 9, "max": 9},
            "macd_long_ratio": {"name": "MACD Long ratio", "type": float, "min": 1, "max": 3, "decimals": 2},
            "macd_short_ratio": {"name": "MACD Short ratio", "type": float, "min": 0.01, "max": 0.1, "decimals": 2},
    },
    "sup_res_cpp": {
            "min_points": {"name": "Min. Points", "type": int, "min": 2, "max": 20},
            "min_diff_points": {"name": "Min. Difference between Points", "type": int, "min": 1, "max": 100},
            "rounding_nb": {"name": "Rounding Number", "type": float, "min": 10, "max": 500, "decimals": 1},
            "ratio": {"name": "WIN/LOSS Ratio", "type": float, "min": 0.5, "max": 5, "decimals": 2},
            "risk": {"name": "Risk per trade", "type": float, "min": 0.005, "max": 0.02, "decimals": 2},
            "leverage": {"name": "Leverage", "type": int, "min": 1, "max": 50},
            "cash": {"name": "Initial balance (USD)", "type": int, "min": 1000, "max": 1000},
            "stop_short": {"name": "Stop short safety", "type": float, "min": 1.0001, "max": 1.01, "decimals": 4},
            "stop_long": {"name": "Stop long safety", "type": float, "min": 0.99, "max": 0.9999, "decimals": 4}

    },
    "bollinger": {
            "window_size": {"name": "window size","type": int, "min": 0.0001, "max": 1000000},
            "num_std": {"name": "deviation number","type": float, "min": 0.0001, "max": 100000}
    },
    "single_index": {
            "test": {"name": "test", "type": int}
    },
    "arima": {
        "start_date": {"name": "start_date", "type": str}
    }
    # "drl": {
    #     "batch_size": {"name": "batch_size", "type": int}
    # }
}

def ms_to_dt(ms: int) -> datetime.datetime:
    return datetime.datetime.utcfromtimestamp(ms / 1000)

def resample_timeframe(data: pd.DataFrame, tf: str) -> pd.DataFrame:
    return data.resample(TF_EQUIV[tf]).agg(
        {"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"}
    )

def get_library():
    lib = CDLL("backtestingCpp/build/libbacktestingCpp.dll", winmode=0)

    # SMA
    lib.Sma_new.restype = c_void_p
    lib.Sma_new.argtypes = [c_char_p, c_char_p, c_char_p, c_longlong, c_longlong]
    lib.Sma_execute_backtest.restype = c_void_p
    lib.Sma_execute_backtest.argtypes = [c_void_p, c_int, c_int]

    lib.Sma_get_pnl.restype = c_double
    lib.Sma_get_pnl.argtypes = [c_void_p]
    lib.Sma_get_max_dd.restype = c_double
    lib.Sma_get_max_dd.argtypes = [c_void_p]


    # PSAR
    lib.Psar_new.restype = c_void_p
    lib.Psar_new.argtypes = [c_char_p, c_char_p, c_char_p, c_longlong, c_longlong]
    lib.Psar_execute_backtest.restype = c_void_p
    lib.Psar_execute_backtest.argtypes = [c_void_p, c_double, c_double, c_double]

    lib.Psar_get_pnl.restype = c_double
    lib.Psar_get_pnl.argtypes = [c_void_p]
    lib.Psar_get_max_dd.restype = c_double
    lib.Psar_get_max_dd.argtypes = [c_void_p]

    # Fractal_test
    lib.fractal_test_new.restype = c_void_p
    lib.fractal_test_new.argtypes = [c_char_p, c_char_p, c_char_p, c_longlong, c_longlong]
    lib.fractal_test_execute_backtest.restype = c_void_p
    lib.fractal_test_execute_backtest.argtypes = [c_void_p, c_int, c_int, c_int, c_int, c_int, c_int, c_double, c_double, c_double, c_int, c_double, c_double, c_double, c_int, c_int, c_int, c_int, c_int, c_double, c_double]

    lib.fractal_test_get_pnl.restype = c_double
    lib.fractal_test_get_pnl.argtypes = [c_void_p]
    lib.fractal_test_get_max_dd.restype = c_double
    lib.fractal_test_get_max_dd.argtypes = [c_void_p]


    # Support Resistance CPP
    lib.sup_res_cpp_new.restype = c_void_p
    lib.sup_res_cpp_new.argtypes = [c_char_p, c_char_p, c_char_p, c_longlong, c_longlong]
    lib.sup_res_cpp_execute_backtest.restype = c_void_p
    lib.sup_res_cpp_execute_backtest.argtypes = [c_void_p, c_int, c_int, c_double, c_double, c_double, c_int, c_int, c_double, c_double]

    lib.sup_res_cpp_get_pnl.restype = c_double
    lib.sup_res_cpp_get_pnl.argtypes = [c_void_p]
    lib.sup_res_cpp_get_daily_pnl.restype = c_double
    lib.sup_res_cpp_get_daily_pnl.argtypes = [c_void_p]
    lib.sup_res_cpp_get_daily_trades.restype = c_int
    lib.sup_res_cpp_get_daily_trades.argtypes = [c_void_p]
    lib.sup_res_cpp_get_zero_per_day.restype = c_int
    lib.sup_res_cpp_get_zero_per_day.argtypes = [c_void_p]

    # Fractal_simple
    lib.fractal_simple_new.restype = c_void_p
    lib.fractal_simple_new.argtypes = [c_char_p, c_char_p, c_char_p, c_longlong, c_longlong]
    lib.fractal_simple_execute_backtest.restype = c_void_p
    lib.fractal_simple_execute_backtest.argtypes = [c_void_p, c_int, c_int, c_int, c_int, c_double,
                                                  c_double, c_double, c_int, c_int]

    lib.fractal_simple_get_pnl.restype = c_double
    lib.fractal_simple_get_pnl.argtypes = [c_void_p]
    lib.fractal_simple_get_max_dd.restype = c_double
    lib.fractal_simple_get_max_dd.argtypes = [c_void_p]


    return lib



