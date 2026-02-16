"""
Utility Functions
Helper functions for formatting, calculations, and display.
"""

import pandas as pd
import numpy as np


def format_large_number(num: float) -> str:
    """Format large numbers with K, M, B suffixes."""
    if num is None or np.isnan(num):
        return "N/A"
    abs_num = abs(num)
    if abs_num >= 1e12:
        return f"${num / 1e12:.2f}T"
    elif abs_num >= 1e9:
        return f"${num / 1e9:.2f}B"
    elif abs_num >= 1e6:
        return f"${num / 1e6:.2f}M"
    elif abs_num >= 1e3:
        return f"${num / 1e3:.2f}K"
    else:
        return f"${num:.2f}"


def format_percentage(value: float, decimals: int = 2) -> str:
    """Format a decimal as percentage string."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"{value * 100:.{decimals}f}%"


def format_currency(value: float, decimals: int = 2) -> str:
    """Format a number as currency."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return "N/A"
    return f"${value:,.{decimals}f}"


def get_color_for_value(value: float) -> str:
    """Return green for positive, red for negative values."""
    if value > 0:
        return "#00d4aa"
    elif value < 0:
        return "#ff4757"
    return "#ffffff"


def calculate_correlation_matrix(data: dict) -> pd.DataFrame:
    """
    Calculate correlation matrix from a dict of DataFrames.

    Parameters
    ----------
    data : dict
        Dictionary mapping ticker -> DataFrame with 'Close' column.

    Returns
    -------
    pd.DataFrame
        Correlation matrix of daily returns.
    """
    returns = pd.DataFrame()
    for ticker, df in data.items():
        if "Daily_Return" in df.columns:
            returns[ticker] = df["Daily_Return"]
        else:
            returns[ticker] = df["Close"].pct_change()

    return returns.corr()


def calculate_portfolio_metrics(
    data: dict, weights: dict = None
) -> dict:
    """
    Calculate basic portfolio-level metrics.

    Parameters
    ----------
    data : dict
        Dictionary mapping ticker -> DataFrame.
    weights : dict, optional
        Dictionary mapping ticker -> weight. Equal-weighted if None.

    Returns
    -------
    dict
        Portfolio metrics.
    """
    tickers = list(data.keys())
    if weights is None:
        w = np.array([1.0 / len(tickers)] * len(tickers))
    else:
        w = np.array([weights.get(t, 0) for t in tickers])
        w = w / w.sum()  # Normalize

    returns = pd.DataFrame()
    for ticker in tickers:
        df = data[ticker]
        returns[ticker] = df["Close"].pct_change()

    returns = returns.dropna()
    portfolio_return = (returns * w).sum(axis=1)

    return {
        "daily_returns": portfolio_return,
        "annual_return": portfolio_return.mean() * 252,
        "annual_volatility": portfolio_return.std() * np.sqrt(252),
        "sharpe_ratio": (portfolio_return.mean() * 252) / (portfolio_return.std() * np.sqrt(252)),
        "correlation_matrix": returns.corr(),
    }
