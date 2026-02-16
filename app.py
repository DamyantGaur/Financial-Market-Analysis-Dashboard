"""
Financial Market Analysis & Forecasting Dashboard
===================================================
A comprehensive Streamlit dashboard for stock market analysis,
technical indicators, risk metrics, and price forecasting.

Author: Damyant Gaur
Tech Stack: Python, Streamlit, Plotly, TensorFlow, Prophet, yfinance
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
import warnings

warnings.filterwarnings("ignore")

# ─── Page Configuration ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="Financial Market Analysis Dashboard",
    page_icon="chart_with_upwards_trend",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Load Google Fonts + custom CSS
st.markdown(
    '<link href="https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap" rel="stylesheet">',
    unsafe_allow_html=True,
)
css_path = os.path.join(os.path.dirname(__file__), "assets", "style.css")
if os.path.exists(css_path):
    with open(css_path) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ─── Import Project Modules ─────────────────────────────────────────────────

from src.data_fetcher import fetch_stock_data, get_stock_info, DEFAULT_TICKERS
from src.technical_analysis import (
    apply_all_indicators,
    compute_rsi,
    compute_macd,
    compute_bollinger_bands,
    compute_sma,
    compute_ema,
)
from src.risk_metrics import (
    compute_daily_returns,
    var_historical,
    var_parametric,
    var_monte_carlo,
    compute_cvar,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    max_drawdown,
    drawdown_series,
    rolling_volatility,
    ewma_volatility,
    annualized_volatility,
    compute_risk_summary,
)
from src.utils import format_large_number, format_percentage, format_currency, get_color_for_value


# ─── Plotly Dark Theme ───────────────────────────────────────────────────────

PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(10, 14, 39, 0)",
    plot_bgcolor="rgba(10, 14, 39, 0.5)",
    font=dict(family="Inter, sans-serif", color="#e6f1ff"),
    xaxis=dict(gridcolor="rgba(100,120,255,0.08)", zerolinecolor="rgba(100,120,255,0.15)"),
    yaxis=dict(gridcolor="rgba(100,120,255,0.08)", zerolinecolor="rgba(100,120,255,0.15)"),
    margin=dict(l=20, r=20, t=40, b=20),
    legend=dict(bgcolor="rgba(10,14,39,0.7)", bordercolor="rgba(100,120,255,0.2)"),
)

COLORS = {
    "primary": "#6366f1",
    "secondary": "#64ffda",
    "positive": "#00d4aa",
    "negative": "#ff4757",
    "warning": "#ffa502",
    "blue": "#4a9eff",
    "purple": "#a855f7",
    "orange": "#ff6b35",
    "candle_up": "#00d4aa",
    "candle_down": "#ff4757",
}


# ─── Sidebar ────────────────────────────────────────────────────────────────

def render_sidebar():
    """Render the sidebar with controls."""
    with st.sidebar:
        st.markdown("## Dashboard Controls")
        st.markdown("---")

        # Ticker selection
        ticker = st.selectbox(
            "Select Stock Ticker",
            options=DEFAULT_TICKERS,
            index=0,
            help="Choose a stock to analyze",
        )

        custom_ticker = st.text_input(
            "Or enter custom ticker",
            placeholder="e.g., NFLX, AMD, BA",
            help="Enter any valid Yahoo Finance ticker",
        )
        if custom_ticker.strip():
            ticker = custom_ticker.strip().upper()

        st.markdown("---")

        # Date range
        st.markdown("### 📅 Date Range")
        col1, col2 = st.columns(2)
        with col1:
            start_date = st.date_input(
                "Start Date",
                value=datetime.today() - timedelta(days=730),
                max_value=datetime.today(),
            )
        with col2:
            end_date = st.date_input(
                "End Date",
                value=datetime.today(),
                max_value=datetime.today(),
            )

        st.markdown("---")

        # Page navigation
        st.markdown("### Navigation")
        page = st.radio(
            "Select Analysis",
            options=[
                "Overview",
                "Technical Analysis",
                "Risk Analysis",
                "Forecasting",
            ],
            label_visibility="collapsed",
        )

        st.markdown("---")
        st.markdown(
            """
            <div style='text-align: center; color: #8892b0; font-size: 0.8rem;'>
                <p>Data: Yahoo Finance</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    return ticker, str(start_date), str(end_date), page


# ─── Page: Overview ─────────────────────────────────────────────────────────

def render_overview(df: pd.DataFrame, ticker: str, info: dict):
    """Render the market overview page."""
    st.markdown(f"## {info.get('name', ticker)} — Market Overview")
    st.markdown(f"**{info.get('sector', '')}** | {info.get('industry', '')}")
    st.markdown("---")

    # ── Key Metrics Row ──
    latest = df.iloc[-1]
    prev = df.iloc[-2] if len(df) > 1 else df.iloc[-1]
    change = latest["Close"] - prev["Close"]
    change_pct = change / prev["Close"]
    returns = compute_daily_returns(df)

    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Current Price", format_currency(latest["Close"]),
                   f"{change_pct:+.2%}")
    with col2:
        st.metric("Day Change", format_currency(change),
                   f"{'▲' if change > 0 else '▼'} {abs(change):.2f}")
    with col3:
        st.metric("52w High / Low",
                   f"{format_currency(info.get('52w_high', df['High'].max()))}",
                   f"Low: {format_currency(info.get('52w_low', df['Low'].min()))}")
    with col4:
        st.metric("Avg Volume",
                   format_large_number(info.get("avg_volume", df["Volume"].mean())))
    with col5:
        st.metric("Market Cap", format_large_number(info.get("market_cap", 0)))

    st.markdown("")

    # ── Candlestick Chart ──
    fig = make_subplots(
        rows=2, cols=1, shared_xaxes=True,
        vertical_spacing=0.03,
        row_heights=[0.75, 0.25],
        subplot_titles=("", "Volume"),
    )

    # Candlestick
    fig.add_trace(
        go.Candlestick(
            x=df.index, open=df["Open"], high=df["High"],
            low=df["Low"], close=df["Close"],
            increasing_line_color=COLORS["candle_up"],
            decreasing_line_color=COLORS["candle_down"],
            increasing_fillcolor=COLORS["candle_up"],
            decreasing_fillcolor=COLORS["candle_down"],
            name="OHLC",
        ),
        row=1, col=1,
    )

    # Add SMA lines
    df_sma = compute_sma(df, periods=[20, 50])
    if "SMA_20" in df_sma.columns:
        fig.add_trace(
            go.Scatter(x=df_sma.index, y=df_sma["SMA_20"],
                       line=dict(color=COLORS["blue"], width=1.5),
                       name="SMA 20", opacity=0.8),
            row=1, col=1,
        )
    if "SMA_50" in df_sma.columns:
        fig.add_trace(
            go.Scatter(x=df_sma.index, y=df_sma["SMA_50"],
                       line=dict(color=COLORS["purple"], width=1.5),
                       name="SMA 50", opacity=0.8),
            row=1, col=1,
        )

    # Volume bars colored by direction
    colors = [COLORS["candle_up"] if c >= o else COLORS["candle_down"]
              for o, c in zip(df["Open"], df["Close"])]
    fig.add_trace(
        go.Bar(x=df.index, y=df["Volume"], marker_color=colors,
               opacity=0.6, name="Volume", showlegend=False),
        row=2, col=1,
    )

    fig.update_layout(
        **PLOTLY_LAYOUT,
        height=600,
        title=f"{ticker} — Price & Volume",
        xaxis_rangeslider_visible=False,
        showlegend=True,
    )
    fig.update_xaxes(gridcolor="rgba(100,120,255,0.08)")
    fig.update_yaxes(gridcolor="rgba(100,120,255,0.08)")

    st.plotly_chart(fig, use_container_width=True)

    # ── Returns Distribution & Cumulative Returns ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Returns Distribution")
        fig_dist = go.Figure()
        fig_dist.add_trace(go.Histogram(
            x=returns, nbinsx=80,
            marker_color=COLORS["primary"],
            opacity=0.75, name="Daily Returns",
        ))
        fig_dist.add_vline(x=returns.mean(), line_dash="dash",
                           line_color=COLORS["secondary"],
                           annotation_text=f"Mean: {returns.mean():.4f}")
        fig_dist.update_layout(**PLOTLY_LAYOUT, height=350,
                               title="Distribution of Daily Returns",
                               xaxis_title="Return", yaxis_title="Frequency")
        st.plotly_chart(fig_dist, use_container_width=True)

    with col2:
        st.markdown("### Cumulative Returns")
        if "Cumulative_Return" in df.columns:
            fig_cum = go.Figure()
            fig_cum.add_trace(go.Scatter(
                x=df.index, y=df["Cumulative_Return"] * 100,
                fill="tozeroy",
                fillcolor="rgba(99, 102, 241, 0.15)",
                line=dict(color=COLORS["primary"], width=2),
                name="Cumulative Return %",
            ))
            fig_cum.update_layout(**PLOTLY_LAYOUT, height=350,
                                  title="Cumulative Returns (%)",
                                  yaxis_title="Return (%)")
            st.plotly_chart(fig_cum, use_container_width=True)

    # ── Company Info ──
    if info.get("description", "N/A") != "N/A":
        with st.expander("About the Company"):
            st.write(info["description"])


# ─── Page: Technical Analysis ────────────────────────────────────────────────

def render_technical_analysis(df: pd.DataFrame, ticker: str):
    """Render the technical analysis page."""
    st.markdown(f"## Technical Analysis — {ticker}")
    st.markdown("---")

    df_ta = apply_all_indicators(df)

    # ── Bollinger Bands ──
    st.markdown("### Bollinger Bands & Moving Averages")
    fig_bb = go.Figure()

    # Bollinger Bands fill
    fig_bb.add_trace(go.Scatter(
        x=df_ta.index, y=df_ta["BB_Upper"],
        line=dict(color="rgba(100,120,255,0.3)", width=1),
        name="BB Upper", showlegend=True,
    ))
    fig_bb.add_trace(go.Scatter(
        x=df_ta.index, y=df_ta["BB_Lower"],
        fill="tonexty", fillcolor="rgba(99,102,241,0.08)",
        line=dict(color="rgba(100,120,255,0.3)", width=1),
        name="BB Lower",
    ))

    # Close price
    fig_bb.add_trace(go.Scatter(
        x=df_ta.index, y=df_ta["Close"],
        line=dict(color="#ffffff", width=1.5),
        name="Close",
    ))

    # SMAs
    for period, color in [(20, COLORS["blue"]), (50, COLORS["purple"]), (200, COLORS["warning"])]:
        col_name = f"SMA_{period}"
        if col_name in df_ta.columns:
            fig_bb.add_trace(go.Scatter(
                x=df_ta.index, y=df_ta[col_name],
                line=dict(color=color, width=1, dash="dot"),
                name=f"SMA {period}", opacity=0.7,
            ))

    fig_bb.update_layout(**PLOTLY_LAYOUT, height=450, title="Price with Bollinger Bands & SMAs")
    st.plotly_chart(fig_bb, use_container_width=True)

    # ── RSI & MACD ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### RSI (Relative Strength Index)")
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(
            x=df_ta.index, y=df_ta["RSI"],
            line=dict(color=COLORS["primary"], width=2),
            name="RSI (14)",
        ))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color=COLORS["negative"],
                          annotation_text="Overbought (70)")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color=COLORS["positive"],
                          annotation_text="Oversold (30)")
        fig_rsi.add_hrect(y0=30, y1=70, fillcolor="rgba(99,102,241,0.05)",
                          line_width=0)
        fig_rsi.update_layout(**PLOTLY_LAYOUT)
        fig_rsi.update_layout(height=350, title="RSI (14-Period)",
                              yaxis=dict(range=[0, 100], gridcolor="rgba(100,120,255,0.08)"))
        st.plotly_chart(fig_rsi, use_container_width=True)

        # RSI interpretation
        current_rsi = df_ta["RSI"].dropna().iloc[-1] if len(df_ta["RSI"].dropna()) > 0 else 50
        if current_rsi > 70:
            st.warning(f"RSI = {current_rsi:.1f} — **Overbought** territory. Potential reversal signal.")
        elif current_rsi < 30:
            st.success(f"RSI = {current_rsi:.1f} — **Oversold** territory. Potential buying opportunity.")
        else:
            st.info(f"RSI = {current_rsi:.1f} — **Neutral** zone.")

    with col2:
        st.markdown("### MACD")
        fig_macd = make_subplots(rows=2, cols=1, shared_xaxes=True,
                                 vertical_spacing=0.05, row_heights=[0.6, 0.4])

        fig_macd.add_trace(go.Scatter(
            x=df_ta.index, y=df_ta["MACD"],
            line=dict(color=COLORS["blue"], width=2), name="MACD",
        ), row=1, col=1)

        fig_macd.add_trace(go.Scatter(
            x=df_ta.index, y=df_ta["MACD_Signal"],
            line=dict(color=COLORS["warning"], width=1.5), name="Signal",
        ), row=1, col=1)

        histogram_colors = [COLORS["positive"] if v >= 0 else COLORS["negative"]
                           for v in df_ta["MACD_Histogram"].fillna(0)]
        fig_macd.add_trace(go.Bar(
            x=df_ta.index, y=df_ta["MACD_Histogram"],
            marker_color=histogram_colors, name="Histogram", opacity=0.6,
        ), row=2, col=1)

        fig_macd.update_layout(**PLOTLY_LAYOUT, height=350, title="MACD (12, 26, 9)")
        fig_macd.update_xaxes(gridcolor="rgba(100,120,255,0.08)")
        fig_macd.update_yaxes(gridcolor="rgba(100,120,255,0.08)")
        st.plotly_chart(fig_macd, use_container_width=True)

        # MACD interpretation
        if len(df_ta["MACD"].dropna()) > 0 and len(df_ta["MACD_Signal"].dropna()) > 0:
            macd_val = df_ta["MACD"].dropna().iloc[-1]
            signal_val = df_ta["MACD_Signal"].dropna().iloc[-1]
            if macd_val > signal_val:
                st.success(f"MACD ({macd_val:.2f}) is **above** Signal ({signal_val:.2f}) — Bullish signal.")
            else:
                st.warning(f"MACD ({macd_val:.2f}) is **below** Signal ({signal_val:.2f}) — Bearish signal.")

    # ── Stochastic Oscillator ──
    st.markdown("### Stochastic Oscillator")
    fig_stoch = go.Figure()
    if "Stoch_K" in df_ta.columns:
        fig_stoch.add_trace(go.Scatter(
            x=df_ta.index, y=df_ta["Stoch_K"],
            line=dict(color=COLORS["primary"], width=2), name="%K",
        ))
        fig_stoch.add_trace(go.Scatter(
            x=df_ta.index, y=df_ta["Stoch_D"],
            line=dict(color=COLORS["warning"], width=1.5, dash="dash"), name="%D",
        ))
        fig_stoch.add_hline(y=80, line_dash="dash", line_color=COLORS["negative"],
                            annotation_text="Overbought")
        fig_stoch.add_hline(y=20, line_dash="dash", line_color=COLORS["positive"],
                            annotation_text="Oversold")
        fig_stoch.update_layout(**PLOTLY_LAYOUT)
        fig_stoch.update_layout(height=300,
                                title="Stochastic Oscillator (%K, %D)",
                                yaxis=dict(range=[0, 100], gridcolor="rgba(100,120,255,0.08)"))
    st.plotly_chart(fig_stoch, use_container_width=True)

    # ── Summary Table ──
    st.markdown("### Current Indicator Values")
    indicators = {}
    if len(df_ta["RSI"].dropna()) > 0:
        indicators["RSI (14)"] = f"{df_ta['RSI'].dropna().iloc[-1]:.2f}"
    if len(df_ta["MACD"].dropna()) > 0:
        indicators["MACD"] = f"{df_ta['MACD'].dropna().iloc[-1]:.4f}"
    if len(df_ta["MACD_Signal"].dropna()) > 0:
        indicators["MACD Signal"] = f"{df_ta['MACD_Signal'].dropna().iloc[-1]:.4f}"
    for period in [20, 50, 200]:
        col_name = f"SMA_{period}"
        if col_name in df_ta.columns and len(df_ta[col_name].dropna()) > 0:
            indicators[f"SMA {period}"] = f"{df_ta[col_name].dropna().iloc[-1]:.2f}"
    if "BB_Upper" in df_ta.columns and len(df_ta["BB_Upper"].dropna()) > 0:
        indicators["BB Upper"] = f"{df_ta['BB_Upper'].dropna().iloc[-1]:.2f}"
        indicators["BB Lower"] = f"{df_ta['BB_Lower'].dropna().iloc[-1]:.2f}"
    if "ATR" in df_ta.columns and len(df_ta["ATR"].dropna()) > 0:
        indicators["ATR (14)"] = f"{df_ta['ATR'].dropna().iloc[-1]:.2f}"

    if indicators:
        ind_df = pd.DataFrame(list(indicators.items()), columns=["Indicator", "Value"])
        st.dataframe(ind_df, use_container_width=True, hide_index=True)


# ─── Page: Risk Analysis ────────────────────────────────────────────────────

def render_risk_analysis(df: pd.DataFrame, ticker: str):
    """Render the risk analysis page."""
    st.markdown(f"## Risk Analysis — {ticker}")
    st.markdown("---")

    returns = compute_daily_returns(df)

    # ── Risk summary metrics ──
    st.markdown("### Key Risk Metrics")
    summary = compute_risk_summary(returns)

    cols = st.columns(4)
    metric_pairs = list(summary.items())
    for i, (key, val) in enumerate(metric_pairs[:8]):
        with cols[i % 4]:
            st.metric(key, val)

    st.markdown("")

    # Additional metrics row
    cols2 = st.columns(4)
    for i, (key, val) in enumerate(metric_pairs[8:12]):
        with cols2[i % 4]:
            st.metric(key, val)

    st.markdown("---")

    # ── VaR Comparison ──
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("### Value at Risk (VaR) Comparison")
        confidence_levels = [0.90, 0.95, 0.99]
        var_data = []
        for conf in confidence_levels:
            var_data.append({
                "Confidence": f"{conf:.0%}",
                "Historical VaR": f"{var_historical(returns, conf):.2%}",
                "Parametric VaR": f"{var_parametric(returns, conf):.2%}",
                "Monte Carlo VaR": f"{var_monte_carlo(returns, conf):.2%}",
                "CVaR (ES)": f"{compute_cvar(returns, conf):.2%}",
            })
        var_df = pd.DataFrame(var_data)
        st.dataframe(var_df, use_container_width=True, hide_index=True)

        st.caption(
            "VaR represents the maximum expected loss at a given confidence level. "
            "CVaR (Expected Shortfall) measures the average loss beyond VaR."
        )

    with col2:
        st.markdown("### VaR Visualization")
        fig_var = go.Figure()
        fig_var.add_trace(go.Histogram(
            x=returns, nbinsx=80, marker_color=COLORS["primary"],
            opacity=0.6, name="Returns",
        ))

        var_95 = var_historical(returns, 0.95)
        var_99 = var_historical(returns, 0.99)
        fig_var.add_vline(x=-var_95, line_dash="dash", line_color=COLORS["warning"],
                          annotation_text=f"VaR 95%: {var_95:.2%}")
        fig_var.add_vline(x=-var_99, line_dash="dash", line_color=COLORS["negative"],
                          annotation_text=f"VaR 99%: {var_99:.2%}")

        fig_var.update_layout(**PLOTLY_LAYOUT, height=400,
                              title="Returns Distribution with VaR Bounds",
                              xaxis_title="Daily Return", yaxis_title="Frequency")
        st.plotly_chart(fig_var, use_container_width=True)

    st.markdown("---")

    # ── Drawdown Chart ──
    st.markdown("### Drawdown Analysis")
    dd = drawdown_series(returns)

    fig_dd = go.Figure()
    fig_dd.add_trace(go.Scatter(
        x=dd.index, y=dd * 100,
        fill="tozeroy", fillcolor="rgba(255, 71, 87, 0.2)",
        line=dict(color=COLORS["negative"], width=1.5),
        name="Drawdown",
    ))
    max_dd = max_drawdown(returns)
    fig_dd.add_hline(y=max_dd * 100, line_dash="dash", line_color=COLORS["negative"],
                     annotation_text=f"Max Drawdown: {max_dd:.2%}")

    fig_dd.update_layout(**PLOTLY_LAYOUT, height=350, title="Underwater (Drawdown) Chart",
                         yaxis_title="Drawdown (%)")
    st.plotly_chart(fig_dd, use_container_width=True)

    # ── Volatility Analysis ──
    st.markdown("### Volatility Analysis")
    col1, col2 = st.columns(2)

    with col1:
        rolling_vol = rolling_volatility(returns, window=21)
        ewma_vol = ewma_volatility(returns, span=21)

        fig_vol = go.Figure()
        fig_vol.add_trace(go.Scatter(
            x=rolling_vol.index, y=rolling_vol * 100,
            line=dict(color=COLORS["primary"], width=2),
            name="Rolling Vol (21d)",
        ))
        fig_vol.add_trace(go.Scatter(
            x=ewma_vol.index, y=ewma_vol * 100,
            line=dict(color=COLORS["secondary"], width=2),
            name="EWMA Vol (21d)",
        ))

        ann_vol = annualized_volatility(returns)
        fig_vol.add_hline(y=ann_vol * 100, line_dash="dot",
                          line_color=COLORS["warning"],
                          annotation_text=f"Avg: {ann_vol:.1%}")

        fig_vol.update_layout(**PLOTLY_LAYOUT, height=350,
                              title="Annualized Rolling Volatility",
                              yaxis_title="Volatility (%)")
        st.plotly_chart(fig_vol, use_container_width=True)

    with col2:
        # Monthly returns heatmap
        st.markdown("#### Monthly Returns Heatmap")
        try:
            # pandas >= 2.2 uses "ME", older uses "M"
            try:
                monthly = df["Close"].resample("ME").last().pct_change()
            except ValueError:
                monthly = df["Close"].resample("M").last().pct_change()

            monthly_df = pd.DataFrame({
                "Year": monthly.index.year,
                "Month": monthly.index.month,
                "Return": monthly.values,
            }).dropna()

            if len(monthly_df) > 0:
                pivot = monthly_df.pivot_table(values="Return", index="Year",
                                               columns="Month", aggfunc="mean")
                month_names = ["Jan", "Feb", "Mar", "Apr", "May", "Jun",
                               "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
                pivot.columns = [month_names[int(c) - 1] for c in pivot.columns]

                fig_heatmap = px.imshow(
                    pivot * 100,
                    color_continuous_scale=["#ff4757", "#1a1a2e", "#00d4aa"],
                    aspect="auto",
                    labels=dict(x="Month", y="Year", color="Return %"),
                )
                fig_heatmap.update_layout(**PLOTLY_LAYOUT, height=300,
                                          title="Monthly Returns (%)")
                st.plotly_chart(fig_heatmap, use_container_width=True)
        except Exception:
            st.info("Not enough data for monthly heatmap.")


# ─── Page: ML Forecasting ───────────────────────────────────────────────────

def render_forecasting(df: pd.DataFrame, ticker: str):
    """Render the forecasting page."""
    st.markdown(f"## Price Forecasting — {ticker}")
    st.markdown("---")

    # ── Model Configuration ──
    st.markdown("### Model Configuration")
    col1, col2, col3 = st.columns(3)

    with col1:
        model_choice = st.selectbox(
            "Select Model",
            options=["LSTM Neural Network", "Prophet", "Both"],
            index=0,
        )
    with col2:
        forecast_days = st.slider("Forecast Days", min_value=7, max_value=90,
                                  value=30, step=7)
    with col3:
        epochs = st.slider("LSTM Epochs", min_value=5, max_value=50,
                          value=15, step=5)

    if st.button("Run Forecast", type="primary", use_container_width=True):

        # ── LSTM ──
        if model_choice in ["LSTM Neural Network", "Both"]:
            st.markdown("---")
            st.markdown("### LSTM Neural Network Forecast")

            with st.spinner("Training LSTM model... This may take a moment."):
                try:
                    from src.forecasting import forecast_lstm
                    result = forecast_lstm(df, epochs=epochs, forecast_days=forecast_days)

                    if "error" in result:
                        st.error(result["error"])
                    else:
                        # Metrics
                        mcols = st.columns(3)
                        with mcols[0]:
                            st.metric("RMSE", f"${result['metrics']['RMSE']:.2f}")
                        with mcols[1]:
                            st.metric("MAE", f"${result['metrics']['MAE']:.2f}")
                        with mcols[2]:
                            st.metric("MAPE", f"{result['metrics']['MAPE']:.2f}%")

                        # Chart
                        fig_lstm = go.Figure()

                        # Actual prices
                        fig_lstm.add_trace(go.Scatter(
                            x=df.index, y=df["Close"],
                            line=dict(color="#ffffff", width=1.5),
                            name="Actual Price", opacity=0.7,
                        ))

                        # Train predictions
                        fig_lstm.add_trace(go.Scatter(
                            x=result["train_predictions"].index,
                            y=result["train_predictions"].values,
                            line=dict(color=COLORS["blue"], width=1.5, dash="dot"),
                            name="Train Prediction",
                        ))

                        # Test predictions
                        fig_lstm.add_trace(go.Scatter(
                            x=result["test_predictions"].index,
                            y=result["test_predictions"].values,
                            line=dict(color=COLORS["positive"], width=2),
                            name="Test Prediction",
                        ))

                        # Future forecast
                        fig_lstm.add_trace(go.Scatter(
                            x=result["future_predictions"].index,
                            y=result["future_predictions"].values,
                            line=dict(color=COLORS["warning"], width=2.5, dash="dash"),
                            name=f"Forecast ({forecast_days}d)",
                        ))

                        # Add confidence zone for future
                        upper = result["future_predictions"].values * 1.05
                        lower = result["future_predictions"].values * 0.95
                        fig_lstm.add_trace(go.Scatter(
                            x=result["future_predictions"].index,
                            y=upper,
                            line=dict(width=0), showlegend=False,
                        ))
                        fig_lstm.add_trace(go.Scatter(
                            x=result["future_predictions"].index,
                            y=lower,
                            fill="tonexty", fillcolor="rgba(255,165,2,0.1)",
                            line=dict(width=0), name="95% Confidence",
                        ))

                        fig_lstm.update_layout(**PLOTLY_LAYOUT, height=500,
                                               title=f"LSTM Price Forecast — {ticker}")
                        st.plotly_chart(fig_lstm, use_container_width=True)

                except Exception as e:
                    st.error(f"LSTM Forecast Error: {str(e)}")
                    st.info("Tip: Make sure TensorFlow is installed: `pip install tensorflow`")

        # ── Prophet ──
        if model_choice in ["Prophet", "Both"]:
            st.markdown("---")
            st.markdown("### Prophet Forecast")

            with st.spinner("Running Prophet forecast..."):
                try:
                    from src.forecasting import forecast_prophet
                    result = forecast_prophet(df, forecast_days=forecast_days)

                    if "error" in result:
                        st.error(result["error"])
                    else:
                        forecast = result["forecast"]

                        # Metrics
                        mcols = st.columns(3)
                        with mcols[0]:
                            st.metric("RMSE", f"${result['metrics']['RMSE']:.2f}")
                        with mcols[1]:
                            st.metric("MAE", f"${result['metrics']['MAE']:.2f}")
                        with mcols[2]:
                            st.metric("MAPE", f"{result['metrics']['MAPE']:.2f}%")

                        # Chart
                        fig_prophet = go.Figure()

                        # Actual
                        fig_prophet.add_trace(go.Scatter(
                            x=df.index, y=df["Close"],
                            line=dict(color="#ffffff", width=1.5),
                            name="Actual", opacity=0.7,
                        ))

                        # Prophet fitted + forecast
                        fig_prophet.add_trace(go.Scatter(
                            x=forecast["ds"], y=forecast["yhat"],
                            line=dict(color=COLORS["primary"], width=2),
                            name="Prophet Forecast",
                        ))

                        # Confidence interval
                        fig_prophet.add_trace(go.Scatter(
                            x=forecast["ds"], y=forecast["yhat_upper"],
                            line=dict(width=0), showlegend=False,
                        ))
                        fig_prophet.add_trace(go.Scatter(
                            x=forecast["ds"], y=forecast["yhat_lower"],
                            fill="tonexty",
                            fillcolor="rgba(99,102,241,0.15)",
                            line=dict(width=0),
                            name="Confidence Interval",
                        ))

                        fig_prophet.update_layout(**PLOTLY_LAYOUT, height=500,
                                                  title=f"Prophet Forecast — {ticker}")
                        st.plotly_chart(fig_prophet, use_container_width=True)

                        # Trend decomposition
                        with st.expander("Trend Decomposition"):
                            col1, col2 = st.columns(2)
                            with col1:
                                fig_trend = go.Figure()
                                fig_trend.add_trace(go.Scatter(
                                    x=forecast["ds"], y=forecast["trend"],
                                    line=dict(color=COLORS["secondary"], width=2),
                                    name="Trend",
                                ))
                                fig_trend.update_layout(**PLOTLY_LAYOUT, height=300,
                                                        title="Overall Trend")
                                st.plotly_chart(fig_trend, use_container_width=True)

                            with col2:
                                if "weekly" in forecast.columns:
                                    fig_weekly = go.Figure()
                                    fig_weekly.add_trace(go.Scatter(
                                        x=forecast["ds"], y=forecast["weekly"],
                                        line=dict(color=COLORS["blue"], width=2),
                                        name="Weekly Seasonality",
                                    ))
                                    fig_weekly.update_layout(**PLOTLY_LAYOUT, height=300,
                                                             title="Weekly Seasonality")
                                    st.plotly_chart(fig_weekly, use_container_width=True)

                except Exception as e:
                    st.error(f"Prophet Forecast Error: {str(e)}")
                    st.info("Tip: Make sure Prophet is installed: `pip install prophet`")

    else:
        st.info("Configure your model parameters and click **Run Forecast** to start.")

        # Show a preview with recent price trend
        st.markdown("### Recent Price Trend")
        recent = df.tail(90)
        fig_recent = go.Figure()
        fig_recent.add_trace(go.Scatter(
            x=recent.index, y=recent["Close"],
            line=dict(color=COLORS["primary"], width=2),
            fill="tozeroy", fillcolor="rgba(99,102,241,0.1)",
            name="Close Price",
        ))
        fig_recent.update_layout(**PLOTLY_LAYOUT, height=300,
                                 title=f"Last 90 Days — {ticker}")
        st.plotly_chart(fig_recent, use_container_width=True)


# ─── Main App ────────────────────────────────────────────────────────────────

def main():
    """Main application entry point."""

    # Header
    st.markdown(
        '<p class="main-header">Financial Market Analysis Dashboard</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="sub-header">Advanced analytics, risk metrics & ML-powered forecasting</p>',
        unsafe_allow_html=True,
    )

    # Sidebar
    ticker, start_date, end_date, page = render_sidebar()

    # Load data
    try:
        with st.spinner(f"Loading data for {ticker}..."):
            df = fetch_stock_data(ticker, start_date, end_date)
            info = get_stock_info(ticker)

        if df is None or len(df) == 0:
            st.error(f"No data available for {ticker}. Please try a different ticker.")
            return

        # Check if using sample data (no cached Yahoo data means sample was generated)
        import os as _os
        cache_dir = _os.path.join(_os.path.dirname(__file__), "sample_data")
        cache_files = [f for f in _os.listdir(cache_dir) if f.startswith(ticker)] if _os.path.isdir(cache_dir) else []
        st.success(f"Loaded {len(df)} trading days for **{ticker}** ({df.index[0].strftime('%Y-%m-%d')} to {df.index[-1].strftime('%Y-%m-%d')})")

    except Exception as e:
        st.error(f"Error loading data: {str(e)}")
        st.info("The dashboard will use generated sample data for demonstration.")
        # Last-resort fallback
        try:
            from src.data_fetcher import _generate_sample_data
            df = _generate_sample_data(ticker, start_date, end_date)
            info = get_stock_info(ticker)
        except Exception:
            st.error("Could not generate data. Please restart the app.")
            return

    # Route to selected page
    if page == "Overview":
        render_overview(df, ticker, info)
    elif page == "Technical Analysis":
        render_technical_analysis(df, ticker)
    elif page == "Risk Analysis":
        render_risk_analysis(df, ticker)
    elif page == "Forecasting":
        render_forecasting(df, ticker)

    # Footer
    st.markdown("---")
    st.markdown(
        """
        <div style='text-align: center; color: #8892b0; padding: 20px;'>
            <p>Financial Market Analysis Dashboard</p>
            <p style='font-size: 0.8rem;'>Data sourced from Yahoo Finance.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
