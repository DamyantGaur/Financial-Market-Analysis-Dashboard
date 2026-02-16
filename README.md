# Financial Market Analysis & Forecasting Dashboard

> A comprehensive, interactive dashboard for stock market analysis, technical indicators, risk assessment, and price forecasting — built with Python, Streamlit & Plotly.

![Python](https://img.shields.io/badge/Python-3.9+-blue?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-1.31-FF4B4B?logo=streamlit&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.15-FF6F00?logo=tensorflow&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

---

## Project Overview

This project provides **end-to-end data science capabilities** applied to financial markets:

- **Data Engineering** — Automated data fetching from Yahoo Finance with local caching
- **Exploratory Analysis** — Candlestick charts, volume analysis, returns distribution
- **Technical Analysis** — RSI, MACD, Bollinger Bands, Stochastic Oscillator, Moving Averages, ATR, OBV, VWAP
- **Risk Management** — Value at Risk (Historical, Parametric, Monte Carlo), Sharpe/Sortino/Calmar Ratios, Max Drawdown, Rolling Volatility
- **Price Forecasting** — LSTM Neural Network & Prophet time-series models

---

## Features

### Market Overview
- Real-time OHLCV candlestick charts with volume overlay
- Key metrics: price, change, market cap, volume
- Returns distribution & cumulative performance analysis
- Company fundamentals at a glance

### Technical Analysis
- **Trend**: SMA (20/50/200), EMA (12/26/50)
- **Momentum**: RSI (14), MACD (12/26/9), Stochastic Oscillator
- **Volatility**: Bollinger Bands (20, 2-sigma), ATR (14)
- **Volume**: OBV, VWAP
- Automated interpretation of indicator signals (overbought/oversold)

### Risk Analysis
- **Value at Risk**: Historical, Parametric (Gaussian), & Monte Carlo simulation methods
- **CVaR / Expected Shortfall**: Tail-risk measurement
- **Performance Ratios**: Sharpe, Sortino, Calmar
- **Drawdown Analysis**: Underwater chart with maximum drawdown tracking
- **Volatility**: Rolling (21-day) & EWMA annualized volatility
- **Monthly Returns Heatmap**: Year x Month performance visualization

### Forecasting
- **LSTM Neural Network**: 3-layer LSTM architecture with dropout regularization
  - Walk-forward train/test validation
  - Future price prediction with confidence bands
- **Prophet**: Bayesian time-series decomposition
  - Trend, weekly & yearly seasonality components
  - Confidence intervals for uncertainty quantification
- **Metrics**: RMSE, MAE, MAPE for model evaluation

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| **Language** | Python 3.9+ |
| **Dashboard** | Streamlit |
| **Visualization** | Plotly (interactive charts) |
| **Data Source** | Yahoo Finance (yfinance) |
| **Deep Learning** | TensorFlow / Keras (LSTM) |
| **Time Series** | Prophet |
| **Scientific Computing** | NumPy, Pandas, SciPy, scikit-learn |
| **Technical Analysis** | Custom implementation + ta library |

---

## Quick Start

### Prerequisites
- Python 3.9 or higher
- pip (Python package manager)

### Installation

```bash
# Clone the repository
git clone https://github.com/DamyantGaur/Financial-Market-Analysis-Dashboard.git
cd Financial-Market-Analysis-Dashboard

# Install dependencies
pip install -r requirements.txt

# Run the dashboard
streamlit run app.py
```

The dashboard will open at `http://localhost:8501`.

### Usage
1. **Select a stock** ticker from the sidebar (or enter a custom one)
2. **Choose a date range** for analysis
3. **Navigate** between the 4 analysis pages using the sidebar radio buttons
4. **On the Forecasting page**, configure model parameters and click "Run Forecast"

---

## Project Structure

```
Financial-Market-Analysis-Dashboard/
├── app.py                        # Main Streamlit dashboard application
├── requirements.txt              # Python dependencies
├── README.md                     # Project documentation
├── assets/
│   └── style.css                 # Custom dark-theme styling
├── src/
│   ├── __init__.py
│   ├── data_fetcher.py           # Yahoo Finance data fetching & caching
│   ├── technical_analysis.py     # Technical indicator computations
│   ├── risk_metrics.py           # Risk & performance metrics
│   ├── forecasting.py            # LSTM & Prophet forecasting models
│   └── utils.py                  # Helper/utility functions
└── sample_data/                  # Cached data (auto-generated)
```

---

## Methodology

### Technical Indicators
- **RSI**: Wilder's smoothing method for momentum measurement
- **MACD**: Dual EMA crossover system with histogram divergence
- **Bollinger Bands**: Statistical volatility bands at plus/minus 2-sigma from 20-day SMA

### Risk Metrics
- **VaR**: Three complementary approaches (historical simulation, parametric Gaussian, Monte Carlo with 10,000 paths)
- **Performance**: Risk-adjusted returns using industry-standard ratios
- **Volatility**: Both rolling window and exponentially weighted methods

### Forecasting Models
- **LSTM**: Sequential model with 3 LSTM layers (50 units each), dropout regularization (0.2), trained with Adam optimizer on MSE loss
- **Prophet**: Additive model with daily/weekly/yearly seasonality, configurable changepoint sensitivity

---

## License

This project is licensed under the MIT License — free for personal and commercial use.

---

## Author

**Damyant Gaur** — Data Scientist & Analyst specializing in Financial Markets
