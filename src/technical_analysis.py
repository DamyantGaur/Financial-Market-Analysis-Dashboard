"""
Technical Analysis Module
Computes popular technical indicators for stock analysis.
"""

import pandas as pd
import numpy as np


def compute_sma(df: pd.DataFrame, column: str = "Close", periods: list = None) -> pd.DataFrame:
    """
    Compute Simple Moving Averages.

    Parameters
    ----------
    df : pd.DataFrame
        OHLCV DataFrame.
    column : str
        Column to compute SMA on.
    periods : list
        List of SMA periods (default: [20, 50, 200]).

    Returns
    -------
    pd.DataFrame
        Original DataFrame with SMA columns added.
    """
    if periods is None:
        periods = [20, 50, 200]
    result = df.copy()
    for period in periods:
        result[f"SMA_{period}"] = result[column].rolling(window=period).mean()
    return result


def compute_ema(df: pd.DataFrame, column: str = "Close", periods: list = None) -> pd.DataFrame:
    """Compute Exponential Moving Averages."""
    if periods is None:
        periods = [12, 26, 50]
    result = df.copy()
    for period in periods:
        result[f"EMA_{period}"] = result[column].ewm(span=period, adjust=False).mean()
    return result


def compute_rsi(df: pd.DataFrame, column: str = "Close", period: int = 14) -> pd.DataFrame:
    """
    Compute Relative Strength Index (RSI).

    The RSI measures the speed and magnitude of recent price changes
    to evaluate overbought or oversold conditions.

    Parameters
    ----------
    period : int
        Lookback period (default: 14).

    Returns
    -------
    pd.DataFrame
        DataFrame with RSI column added.
    """
    result = df.copy()
    delta = result[column].diff()

    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)

    avg_gain = gain.rolling(window=period, min_periods=period).mean()
    avg_loss = loss.rolling(window=period, min_periods=period).mean()

    # Use Wilder's smoothing after initial SMA
    for i in range(period, len(avg_gain)):
        avg_gain.iloc[i] = (avg_gain.iloc[i - 1] * (period - 1) + gain.iloc[i]) / period
        avg_loss.iloc[i] = (avg_loss.iloc[i - 1] * (period - 1) + loss.iloc[i]) / period

    rs = avg_gain / avg_loss
    result["RSI"] = 100 - (100 / (1 + rs))

    return result


def compute_macd(
    df: pd.DataFrame,
    column: str = "Close",
    fast: int = 12,
    slow: int = 26,
    signal: int = 9,
) -> pd.DataFrame:
    """
    Compute MACD (Moving Average Convergence Divergence).

    Parameters
    ----------
    fast : int
        Fast EMA period (default: 12).
    slow : int
        Slow EMA period (default: 26).
    signal : int
        Signal line EMA period (default: 9).

    Returns
    -------
    pd.DataFrame
        DataFrame with MACD, MACD_Signal, and MACD_Histogram columns.
    """
    result = df.copy()
    ema_fast = result[column].ewm(span=fast, adjust=False).mean()
    ema_slow = result[column].ewm(span=slow, adjust=False).mean()

    result["MACD"] = ema_fast - ema_slow
    result["MACD_Signal"] = result["MACD"].ewm(span=signal, adjust=False).mean()
    result["MACD_Histogram"] = result["MACD"] - result["MACD_Signal"]

    return result


def compute_bollinger_bands(
    df: pd.DataFrame, column: str = "Close", period: int = 20, std_dev: float = 2.0
) -> pd.DataFrame:
    """
    Compute Bollinger Bands.

    Parameters
    ----------
    period : int
        Lookback period for the moving average (default: 20).
    std_dev : float
        Number of standard deviations for the bands (default: 2.0).

    Returns
    -------
    pd.DataFrame
        DataFrame with BB_Middle, BB_Upper, BB_Lower, and BB_Width columns.
    """
    result = df.copy()
    result["BB_Middle"] = result[column].rolling(window=period).mean()
    rolling_std = result[column].rolling(window=period).std()

    result["BB_Upper"] = result["BB_Middle"] + (rolling_std * std_dev)
    result["BB_Lower"] = result["BB_Middle"] - (rolling_std * std_dev)
    result["BB_Width"] = (result["BB_Upper"] - result["BB_Lower"]) / result["BB_Middle"]

    return result


def compute_obv(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute On-Balance Volume (OBV).

    OBV measures buying and selling pressure as a cumulative indicator,
    adding volume on up days and subtracting on down days.
    """
    result = df.copy()
    obv = [0]
    for i in range(1, len(result)):
        if result["Close"].iloc[i] > result["Close"].iloc[i - 1]:
            obv.append(obv[-1] + result["Volume"].iloc[i])
        elif result["Close"].iloc[i] < result["Close"].iloc[i - 1]:
            obv.append(obv[-1] - result["Volume"].iloc[i])
        else:
            obv.append(obv[-1])
    result["OBV"] = obv
    return result


def compute_vwap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Volume Weighted Average Price (VWAP).
    """
    result = df.copy()
    typical_price = (result["High"] + result["Low"] + result["Close"]) / 3
    result["VWAP"] = (typical_price * result["Volume"]).cumsum() / result["Volume"].cumsum()
    return result


def compute_atr(df: pd.DataFrame, period: int = 14) -> pd.DataFrame:
    """
    Compute Average True Range (ATR).
    """
    result = df.copy()
    high_low = result["High"] - result["Low"]
    high_close = np.abs(result["High"] - result["Close"].shift())
    low_close = np.abs(result["Low"] - result["Close"].shift())

    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    result["ATR"] = true_range.rolling(window=period).mean()
    return result


def compute_stochastic(
    df: pd.DataFrame, k_period: int = 14, d_period: int = 3
) -> pd.DataFrame:
    """Compute Stochastic Oscillator (%K and %D)."""
    result = df.copy()
    low_min = result["Low"].rolling(window=k_period).min()
    high_max = result["High"].rolling(window=k_period).max()

    result["Stoch_K"] = 100 * (result["Close"] - low_min) / (high_max - low_min)
    result["Stoch_D"] = result["Stoch_K"].rolling(window=d_period).mean()

    return result


def apply_all_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply all technical indicators to a DataFrame.
    This is a convenience function for the dashboard.
    """
    result = df.copy()
    result = compute_sma(result)
    result = compute_ema(result)
    result = compute_rsi(result)
    result = compute_macd(result)
    result = compute_bollinger_bands(result)
    result = compute_obv(result)
    result = compute_vwap(result)
    result = compute_atr(result)
    result = compute_stochastic(result)
    return result
