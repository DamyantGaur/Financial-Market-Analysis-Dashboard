"""
Risk Metrics Module
Computes key risk and performance metrics for portfolio/stock analysis.
"""

import pandas as pd
import numpy as np
from scipy import stats


def compute_daily_returns(df: pd.DataFrame, column: str = "Close") -> pd.Series:
    """Compute daily percentage returns."""
    return df[column].pct_change().dropna()


def compute_log_returns(df: pd.DataFrame, column: str = "Close") -> pd.Series:
    """Compute daily log returns."""
    return np.log(df[column] / df[column].shift(1)).dropna()


# ─── Value at Risk (VaR) ────────────────────────────────────────────────────

def var_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Historical Value at Risk.

    Parameters
    ----------
    returns : pd.Series
        Daily returns series.
    confidence : float
        Confidence level (default: 0.95 = 95%).

    Returns
    -------
    float
        VaR as a positive number representing the potential loss.
    """
    return -np.percentile(returns.dropna(), (1 - confidence) * 100)


def var_parametric(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Parametric (Variance-Covariance) Value at Risk.
    Assumes normally distributed returns.
    """
    mean = returns.mean()
    std = returns.std()
    z_score = stats.norm.ppf(1 - confidence)
    return -(mean + z_score * std)


def var_monte_carlo(
    returns: pd.Series,
    confidence: float = 0.95,
    n_simulations: int = 10000,
    n_days: int = 1,
) -> float:
    """
    Monte Carlo Value at Risk.
    Simulates future returns using historical mean and standard deviation.
    """
    mean = returns.mean()
    std = returns.std()
    simulated = np.random.normal(mean, std, (n_simulations, n_days))
    portfolio_returns = simulated.sum(axis=1)
    return -np.percentile(portfolio_returns, (1 - confidence) * 100)


def compute_cvar(returns: pd.Series, confidence: float = 0.95) -> float:
    """
    Conditional VaR (Expected Shortfall).
    Average of losses beyond the VaR threshold.
    """
    var = var_historical(returns, confidence)
    tail = returns[returns <= -var]
    if len(tail) == 0:
        return var  # fallback to VaR if no tail losses
    return -tail.mean()


# ─── Performance Ratios ─────────────────────────────────────────────────────

def sharpe_ratio(
    returns: pd.Series, risk_free_rate: float = 0.02, trading_days: int = 252
) -> float:
    """
    Compute annualized Sharpe Ratio.

    Parameters
    ----------
    returns : pd.Series
        Daily returns.
    risk_free_rate : float
        Annual risk-free rate (default: 2%).
    trading_days : int
        Number of trading days per year.

    Returns
    -------
    float
        Annualized Sharpe Ratio.
    """
    daily_rf = risk_free_rate / trading_days
    excess_returns = returns - daily_rf
    if excess_returns.std() == 0:
        return 0.0
    return np.sqrt(trading_days) * excess_returns.mean() / excess_returns.std()


def sortino_ratio(
    returns: pd.Series, risk_free_rate: float = 0.02, trading_days: int = 252
) -> float:
    """
    Compute annualized Sortino Ratio.
    Uses only downside deviation instead of total standard deviation.
    """
    daily_rf = risk_free_rate / trading_days
    excess_returns = returns - daily_rf
    downside = excess_returns[excess_returns < 0]
    if len(downside) == 0 or downside.std() == 0:
        return 0.0
    return np.sqrt(trading_days) * excess_returns.mean() / downside.std()


def calmar_ratio(returns: pd.Series, trading_days: int = 252) -> float:
    """
    Compute Calmar Ratio.
    Annualized return divided by maximum drawdown.
    """
    annual_return = returns.mean() * trading_days
    max_dd = max_drawdown(returns)
    if max_dd == 0:
        return 0.0
    return annual_return / abs(max_dd)


# ─── Drawdown Analysis ──────────────────────────────────────────────────────

def max_drawdown(returns: pd.Series) -> float:
    """
    Compute Maximum Drawdown.

    Returns
    -------
    float
        Maximum drawdown as a negative decimal (e.g., -0.25 = 25% drawdown).
    """
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    drawdown = (cumulative - rolling_max) / rolling_max
    return drawdown.min()


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Compute the full drawdown series."""
    cumulative = (1 + returns).cumprod()
    rolling_max = cumulative.cummax()
    return (cumulative - rolling_max) / rolling_max


# ─── Volatility ─────────────────────────────────────────────────────────────

def rolling_volatility(
    returns: pd.Series, window: int = 21, trading_days: int = 252
) -> pd.Series:
    """
    Compute annualized rolling volatility.

    Parameters
    ----------
    window : int
        Rolling window size in trading days (default: 21 ≈ 1 month).
    trading_days : int
        Trading days per year for annualization.

    Returns
    -------
    pd.Series
        Annualized rolling volatility.
    """
    return returns.rolling(window=window).std() * np.sqrt(trading_days)


def ewma_volatility(
    returns: pd.Series, span: int = 21, trading_days: int = 252
) -> pd.Series:
    """
    Compute EWMA (Exponentially Weighted) annualized volatility.
    Gives more weight to recent observations.
    """
    return returns.ewm(span=span).std() * np.sqrt(trading_days)


def annualized_volatility(returns: pd.Series, trading_days: int = 252) -> float:
    """Compute total annualized volatility."""
    return returns.std() * np.sqrt(trading_days)


# ─── Summary Report ─────────────────────────────────────────────────────────

def compute_risk_summary(returns: pd.Series, risk_free_rate: float = 0.02) -> dict:
    """
    Compute a comprehensive risk summary.

    Returns
    -------
    dict
        Dictionary with all key risk metrics.
    """
    return {
        "Annualized Return": f"{returns.mean() * 252:.2%}",
        "Annualized Volatility": f"{annualized_volatility(returns):.2%}",
        "Sharpe Ratio": f"{sharpe_ratio(returns, risk_free_rate):.3f}",
        "Sortino Ratio": f"{sortino_ratio(returns, risk_free_rate):.3f}",
        "Calmar Ratio": f"{calmar_ratio(returns):.3f}",
        "Max Drawdown": f"{max_drawdown(returns):.2%}",
        "VaR (95% Historical)": f"{var_historical(returns, 0.95):.2%}",
        "VaR (99% Historical)": f"{var_historical(returns, 0.99):.2%}",
        "CVaR (95%)": f"{compute_cvar(returns, 0.95):.2%}",
        "Skewness": f"{returns.skew():.3f}",
        "Kurtosis": f"{returns.kurtosis():.3f}",
        "Best Day": f"{returns.max():.2%}",
        "Worst Day": f"{returns.min():.2%}",
        "Positive Days": f"{(returns > 0).sum() / len(returns):.1%}",
    }
