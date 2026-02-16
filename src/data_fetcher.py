"""
Data Fetcher Module
Fetches historical stock market data via Yahoo Finance with local caching.
Includes built-in sample data generation for offline/demo usage.
"""

import os
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import time

CACHE_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "sample_data")


def ensure_cache_dir():
    """Create cache directory if it doesn't exist."""
    os.makedirs(CACHE_DIR, exist_ok=True)


def get_cache_path(ticker: str, start: str, end: str) -> str:
    """Generate a cache file path for a given ticker and date range."""
    safe_name = f"{ticker}_{start}_{end}.csv"
    return os.path.join(CACHE_DIR, safe_name)


def _generate_sample_data(ticker: str, start_date: str, end_date: str) -> pd.DataFrame:
    """
    Generate realistic synthetic stock data for demo/offline usage.
    Uses random walk with drift to simulate realistic price movements.
    """
    # Seed based on ticker name for consistency
    seed = sum(ord(c) for c in ticker) % 10000
    rng = np.random.RandomState(seed)

    # Ticker-specific starting prices and volatility
    ticker_params = {
        "AAPL": {"price": 185.0, "vol": 0.018, "drift": 0.0003},
        "MSFT": {"price": 415.0, "vol": 0.017, "drift": 0.0004},
        "GOOGL": {"price": 175.0, "vol": 0.019, "drift": 0.0003},
        "AMZN": {"price": 195.0, "vol": 0.021, "drift": 0.0003},
        "TSLA": {"price": 245.0, "vol": 0.032, "drift": 0.0001},
        "META": {"price": 530.0, "vol": 0.022, "drift": 0.0004},
        "NVDA": {"price": 880.0, "vol": 0.028, "drift": 0.0005},
        "JPM": {"price": 198.0, "vol": 0.014, "drift": 0.0002},
    }

    params = ticker_params.get(ticker, {"price": 100.0, "vol": 0.02, "drift": 0.0002})
    base_price = params["price"]
    daily_vol = params["vol"]
    daily_drift = params["drift"]

    # Generate business day date range
    dates = pd.bdate_range(start=start_date, end=end_date)
    n = len(dates)
    if n == 0:
        dates = pd.bdate_range(end=end_date, periods=500)
        n = len(dates)

    # Geometric Brownian Motion for realistic price simulation
    returns = rng.normal(daily_drift, daily_vol, n)

    # Add some mean-reversion and momentum patterns
    for i in range(5, n):
        # Mean reversion component
        returns[i] += -0.02 * (sum(returns[i - 5 : i]) / 5)
        # Occasional larger moves (earnings, news)
        if rng.random() < 0.02:
            returns[i] *= rng.choice([2.5, -2.5])

    prices = base_price * np.exp(np.cumsum(returns))

    # Generate OHLCV data
    daily_ranges = rng.uniform(0.005, 0.025, n)
    volumes = rng.lognormal(mean=np.log(50_000_000), sigma=0.4, size=n).astype(int)

    df = pd.DataFrame(index=dates)
    df.index.name = "Date"
    df["Close"] = prices
    df["Open"] = prices * (1 + rng.uniform(-0.005, 0.005, n))
    df["High"] = np.maximum(df["Open"], df["Close"]) * (1 + daily_ranges)
    df["Low"] = np.minimum(df["Open"], df["Close"]) * (1 - daily_ranges)
    df["Volume"] = volumes

    # Ensure OHLC consistency
    df["High"] = df[["Open", "High", "Close"]].max(axis=1)
    df["Low"] = df[["Open", "Low", "Close"]].min(axis=1)

    # Add computed columns
    df["Daily_Return"] = df["Close"].pct_change()
    df["Cumulative_Return"] = (1 + df["Daily_Return"]).cumprod() - 1
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    return df


def fetch_stock_data(
    ticker: str,
    start_date: str = None,
    end_date: str = None,
    period: str = "2y",
    use_cache: bool = True,
) -> pd.DataFrame:
    """
    Fetch historical OHLCV data for a given ticker.

    Tries Yahoo Finance first, falls back to cached data or sample data.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. 'AAPL', 'MSFT').
    start_date : str, optional
        Start date in 'YYYY-MM-DD' format.
    end_date : str, optional
        End date in 'YYYY-MM-DD' format.
    period : str, optional
        Period to fetch if dates are not specified (default '2y').
    use_cache : bool
        Whether to cache/load from local CSV files.

    Returns
    -------
    pd.DataFrame
        DataFrame with columns: Open, High, Low, Close, Volume, plus computed fields.
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y-%m-%d")
    if start_date is None:
        start_date = (datetime.today() - timedelta(days=730)).strftime("%Y-%m-%d")

    ensure_cache_dir()
    cache_path = get_cache_path(ticker, start_date, end_date)

    # Try loading from cache first
    if use_cache and os.path.exists(cache_path):
        try:
            df = pd.read_csv(cache_path, index_col=0, parse_dates=True)
            if len(df) > 10:
                return df
        except Exception:
            pass

    # Try fetching from Yahoo Finance
    df = _try_yahoo_finance(ticker, start_date, end_date, period)

    if df is not None and len(df) > 0:
        # Cache the data
        if use_cache:
            try:
                df.to_csv(cache_path)
            except Exception:
                pass
        return df

    # Fallback: generate sample data for demo
    print(f"[INFO] Using generated sample data for {ticker} (Yahoo Finance unavailable)")
    df = _generate_sample_data(ticker, start_date, end_date)

    # Cache the sample data too
    if use_cache:
        try:
            df.to_csv(cache_path)
        except Exception:
            pass

    return df


def _try_yahoo_finance(ticker, start_date, end_date, period):
    """Attempt to fetch data from Yahoo Finance with retries."""
    try:
        import yfinance as yf
    except ImportError:
        return None

    # Try multiple approaches
    approaches = [
        lambda: yf.download(ticker, start=start_date, end=end_date, progress=False, auto_adjust=False),
        lambda: yf.Ticker(ticker).history(start=start_date, end=end_date, auto_adjust=False),
        lambda: yf.download(ticker, period=period, progress=False, auto_adjust=False),
        lambda: yf.Ticker(ticker).history(period=period),
    ]

    for attempt_fn in approaches:
        try:
            df = attempt_fn()
            if df is not None and len(df) > 0:
                # Standardize columns
                df = _standardize_columns(df)
                if df is not None and len(df) > 0:
                    return df
        except Exception:
            time.sleep(0.5)
            continue

    return None


def _standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize DataFrame columns regardless of yfinance output format."""
    if df is None or len(df) == 0:
        return None

    # Handle multi-level columns (from yf.download with single ticker)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)

    # Ensure we have the required columns
    required = ["Open", "High", "Low", "Close", "Volume"]
    # Try case-insensitive match
    col_map = {}
    for req in required:
        for col in df.columns:
            if str(col).lower() == req.lower():
                col_map[col] = req
                break

    if len(col_map) < 4:  # Need at least OHLC
        return None

    df = df.rename(columns=col_map)

    # Keep only needed columns
    keep_cols = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep_cols]
    df = df.dropna()

    if len(df) == 0:
        return None

    # Add computed columns
    df["Daily_Return"] = df["Close"].pct_change()
    df["Cumulative_Return"] = (1 + df["Daily_Return"]).cumprod() - 1
    df["Log_Return"] = np.log(df["Close"] / df["Close"].shift(1))

    df.index.name = "Date"
    return df


def fetch_multiple_stocks(
    tickers: list, start_date: str = None, end_date: str = None
) -> dict:
    """
    Fetch data for multiple tickers.

    Returns
    -------
    dict
        Dictionary mapping ticker -> DataFrame.
    """
    data = {}
    for ticker in tickers:
        try:
            data[ticker] = fetch_stock_data(ticker, start_date, end_date)
        except Exception as e:
            print(f"Warning: Could not fetch {ticker}: {e}")
    return data


def get_stock_info(ticker: str) -> dict:
    """Fetch company information for a ticker. Falls back to defaults."""
    # Default info for known tickers
    default_info = {
        "AAPL": {"name": "Apple Inc.", "sector": "Technology", "industry": "Consumer Electronics",
                 "market_cap": 2_900_000_000_000, "pe_ratio": 30.5, "52w_high": 237.0, "52w_low": 164.0,
                 "avg_volume": 55_000_000,
                 "description": "Apple Inc. designs, manufactures, and markets smartphones, personal computers, tablets, wearables, and accessories worldwide. The company offers iPhone, Mac, iPad, and wearables, as well as services including advertising, AppleCare, cloud, digital content, payment, and other services."},
        "MSFT": {"name": "Microsoft Corporation", "sector": "Technology", "industry": "Software—Infrastructure",
                 "market_cap": 3_100_000_000_000, "pe_ratio": 36.2, "52w_high": 468.0, "52w_low": 362.0,
                 "avg_volume": 22_000_000,
                 "description": "Microsoft Corporation develops and supports software, services, devices, and solutions worldwide. Its products include operating systems, cross-device productivity and collaboration applications, server applications, business solution applications, desktop and server management tools, and video games."},
        "GOOGL": {"name": "Alphabet Inc.", "sector": "Technology", "industry": "Internet Content & Information",
                  "market_cap": 2_200_000_000_000, "pe_ratio": 25.8, "52w_high": 201.0, "52w_low": 131.0,
                  "avg_volume": 25_000_000,
                  "description": "Alphabet Inc. offers various products and platforms in the United States, Europe, the Middle East, Africa, the Asia-Pacific, Canada, and Latin America. It operates through Google Services, Google Cloud, and Other Bets segments."},
        "AMZN": {"name": "Amazon.com, Inc.", "sector": "Consumer Cyclical", "industry": "Internet Retail",
                 "market_cap": 2_000_000_000_000, "pe_ratio": 62.1, "52w_high": 233.0, "52w_low": 151.0,
                 "avg_volume": 45_000_000,
                 "description": "Amazon.com, Inc. engages in the retail sale of consumer products, advertising, and subscription services through online and physical stores worldwide."},
        "TSLA": {"name": "Tesla, Inc.", "sector": "Consumer Cyclical", "industry": "Auto Manufacturers",
                 "market_cap": 780_000_000_000, "pe_ratio": 65.3, "52w_high": 358.0, "52w_low": 138.0,
                 "avg_volume": 95_000_000,
                 "description": "Tesla, Inc. designs, develops, manufactures, leases, and sells electric vehicles, and energy generation and storage systems worldwide."},
        "META": {"name": "Meta Platforms, Inc.", "sector": "Technology", "industry": "Internet Content & Information",
                 "market_cap": 1_400_000_000_000, "pe_ratio": 33.8, "52w_high": 602.0, "52w_low": 390.0,
                 "avg_volume": 15_000_000,
                 "description": "Meta Platforms, Inc. engages in the development of products that enable people to connect and share with friends and family through mobile devices, personal computers, virtual reality headsets, and wearables worldwide."},
        "NVDA": {"name": "NVIDIA Corporation", "sector": "Technology", "industry": "Semiconductors",
                 "market_cap": 3_400_000_000_000, "pe_ratio": 68.4, "52w_high": 974.0, "52w_low": 473.0,
                 "avg_volume": 40_000_000,
                 "description": "NVIDIA Corporation provides graphics and compute and networking solutions worldwide. Its products are used in gaming, professional visualization, data center, and automotive markets."},
        "JPM": {"name": "JPMorgan Chase & Co.", "sector": "Financial Services", "industry": "Banks—Diversified",
                "market_cap": 580_000_000_000, "pe_ratio": 12.1, "52w_high": 240.0, "52w_low": 182.0,
                "avg_volume": 10_000_000,
                "description": "JPMorgan Chase & Co. operates as a financial services company worldwide. It operates through four segments: Consumer & Community Banking, Corporate & Investment Bank, Commercial Banking, and Asset & Wealth Management."},
    }

    info = default_info.get(ticker, {})

    # Try fetching live info (non-blocking)
    try:
        import yfinance as yf
        stock = yf.Ticker(ticker)
        live_info = stock.info
        if live_info and isinstance(live_info, dict) and live_info.get("longName"):
            return {
                "name": live_info.get("longName", info.get("name", ticker)),
                "sector": live_info.get("sector", info.get("sector", "N/A")),
                "industry": live_info.get("industry", info.get("industry", "N/A")),
                "market_cap": live_info.get("marketCap", info.get("market_cap", 0)),
                "pe_ratio": live_info.get("trailingPE", info.get("pe_ratio", 0)),
                "dividend_yield": live_info.get("dividendYield", 0),
                "52w_high": live_info.get("fiftyTwoWeekHigh", info.get("52w_high", 0)),
                "52w_low": live_info.get("fiftyTwoWeekLow", info.get("52w_low", 0)),
                "avg_volume": live_info.get("averageVolume", info.get("avg_volume", 0)),
                "description": live_info.get("longBusinessSummary", info.get("description", "N/A")),
            }
    except Exception:
        pass

    # Return default info
    return {
        "name": info.get("name", ticker),
        "sector": info.get("sector", "N/A"),
        "industry": info.get("industry", "N/A"),
        "market_cap": info.get("market_cap", 0),
        "pe_ratio": info.get("pe_ratio", 0),
        "dividend_yield": 0,
        "52w_high": info.get("52w_high", 0),
        "52w_low": info.get("52w_low", 0),
        "avg_volume": info.get("avg_volume", 0),
        "description": info.get("description", "N/A"),
    }


# Default tickers for quick demo
DEFAULT_TICKERS = ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "JPM"]
