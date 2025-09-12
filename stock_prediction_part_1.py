# app.py
# -------------------------------------------------------
# Advanced Multi-Stock Analytics Dashboard (Yahoo Finance)
# Phases 1–3: Data Collection, Understanding, Visualization
# -------------------------------------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime, timedelta
from typing import Dict, List

# -------------------------
# Page / Style
# -------------------------
st.set_page_config(page_title="Multi-Stock Analytics Dashboard", layout="wide")
st.title("📊 Multi-Stock Analytics Dashboard (Yahoo Finance)")
st.caption("Phases 1–3: Data Collection · Data Understanding · Data Visualization")

# -------------------------
# Helpers
# -------------------------
SECTOR_BUCKETS = {
    "Tech": ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA", "META", "NVDA", "NFLX"],
    "Banking/Finance": ["JPM", "GS", "BAC", "RY", "TD", "C", "WFC", "MS"],
    "Energy": ["XOM", "CVX", "COP", "SU", "EOG"],
    "Consumer": ["PG", "KO", "PEP", "MCD", "COST"]
}

def _validate_tickers(tickers: List[str]) -> List[str]:
    # Remove blanks/duplicates and uppercase everything
    clean = []
    for t in tickers:
        if isinstance(t, str):
            t = t.strip().upper()
            if t and t not in clean:
                clean.append(t)
    return clean

@st.cache_data(show_spinner=False)
def fetch_multi(tickers: List[str], start: datetime, end: datetime, interval: str) -> Dict[str, pd.DataFrame]:
    """
    Fetch OHLCV for multiple tickers with yfinance in a single call.
    Returns dict[ticker] -> DataFrame(OHLCV, Adj Close, Volume) indexed by Date.
    """
    if not tickers:
        return {}
    df = yf.download(tickers, start=start, end=end, interval=interval, group_by="ticker", auto_adjust=False, threads=True, progress=False)
    # Normalize to dict of single-index DataFrames
    out: Dict[str, pd.DataFrame] = {}
    if isinstance(df.columns, pd.MultiIndex):
        for t in tickers:
            if (t,) in [(lvl,) for lvl in df.columns.levels[0]]:
                try:
                    sub = df[t].dropna(how="all")
                    if not sub.empty:
                        out[t] = sub.copy()
                except KeyError:
                    pass
    else:
        # Single ticker case: columns are flat
        t = tickers[0]
        if not df.empty:
            out[t] = df.copy()
    # Ensure Date index is DatetimeIndex and sorted
    for t, d in list(out.items()):
        d = d.copy()
        d.index = pd.to_datetime(d.index)
        d.sort_index(inplace=True)
        out[t] = d
    return out

def build_adj_close_frame(data_dict: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    frames = []
    cols = []
    for t, d in data_dict.items():
        if "Adj Close" in d.columns:
            frames.append(d["Adj Close"].rename(t))
            cols.append(t)
    if not frames:
        return pd.DataFrame()
    adj = pd.concat(frames, axis=1)
    adj.columns = cols
    return adj.dropna(how="all")

def daily_returns(prices: pd.DataFrame) -> pd.DataFrame:
    if prices.empty:
        return prices
    return prices.pct_change().dropna(how="all")

def compute_indicators(df: pd.DataFrame, ma_short: int = 50, ma_long: int = 200) -> pd.DataFrame:
    """
    Adds SMA/EMA, Bollinger Bands, RSI to a per-ticker OHLCV DF.
    """
    d = df.copy()
    close = d["Adj Close"] if "Adj Close" in d.columns else d["Close"]
    d["SMA_short"] = close.rolling(ma_short, min_periods=ma_short//2).mean()
    d["SMA_long"]  = close.rolling(ma_long,  min_periods=ma_long//2).mean()
    d["EMA_20"] = close.ewm(span=20, adjust=False).mean()
    # Bollinger (20, 2)
    bb_mid = close.rolling(20, min_periods=10).mean()
    bb_std = close.rolling(20, min_periods=10).std()
    d["BB_MID"] = bb_mid
    d["BB_UP"]  = bb_mid + 2 * bb_std
    d["BB_LOW"] = bb_mid - 2 * bb_std
    # RSI(14)
    delta = close.diff()
    gain = np.where(delta > 0, delta, 0.0)
    loss = np.where(delta < 0, -delta, 0.0)
    roll_up = pd.Series(gain, index=close.index).ewm(span=14, adjust=False).mean()
    roll_down = pd.Series(loss, index=close.index).ewm(span=14, adjust=False).mean()
    rs = roll_up / (roll_down.replace(0, np.nan))
    d["RSI14"] = 100 - (100 / (1 + rs))
    return d

def to_portfolio(returns_df: pd.DataFrame, weights: pd.Series) -> pd.Series:
    """Compute portfolio daily returns from component returns and weights."""
    aligned = returns_df[weights.index]
    return (aligned * weights.values).sum(axis=1)

def ann_metrics(ret: pd.Series, trading_days: int = 252) -> dict:
    if ret.empty:
        return {"ann_return": np.nan, "ann_vol": np.nan, "sharpe": np.nan}
    mean_daily = ret.mean()
    std_daily = ret.std()
    ann_return = (1 + mean_daily) ** trading_days - 1
    ann_vol = std_daily * np.sqrt(trading_days)
    sharpe = np.nan if ann_vol == 0 else (ann_return / ann_vol)
    return {"ann_return": ann_return, "ann_vol": ann_vol, "sharpe": sharpe}

def normalize_100(prices: pd.DataFrame) -> pd.DataFrame:
    """Normalize prices to start at 100 for comparison."""
    base = prices.iloc[0]
    return prices.divide(base).multiply(100)

# -------------------------
# Sidebar Controls
# -------------------------
st.sidebar.header("Controls")

# Quick-pick sectors
with st.sidebar.expander("Quick Picks by Sector"):
    selected_buckets = []
    for bucket, symbols in SECTOR_BUCKETS.items():
        if st.checkbox(bucket, value=(bucket == "Tech")):
            selected_buckets.extend(symbols)

manual_tickers = st.sidebar.text_input("Add/override tickers (comma-separated)", "")
tickers = _validate_tickers(list(set(selected_buckets + [t.strip() for t in manual_tickers.split(",") if t.strip()])))

if not tickers:
    tickers = ["AAPL", "MSFT", "JPM", "RY"]  # sensible defaults

col1, col2 = st.sidebar.columns(2)
with col1:
    end_date = st.date_input("End Date", datetime.today())
with col2:
    start_date = st.date_input("Start Date", datetime.today() - timedelta(days=365*3))

interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"], index=0)

ma_short = st.sidebar.slider("Short SMA (days)", 5, 100, 50)
ma_long  = st.sidebar.slider("Long SMA (days)", 50, 300, 200)
show_downloads = st.sidebar.checkbox("Enable CSV downloads", value=True)

run = st.sidebar.button("Fetch & Analyze")

# -------------------------
# Data Fetch + Guards
# -------------------------
if run:
    st.toast(f"Fetching: {', '.join(tickers)}", icon="🔎")
    data_dict = fetch_multi(tickers, pd.to_datetime(start_date), pd.to_datetime(end_date), interval)
    valid = sorted([t for t, df in data_dict.items() if not df.empty])
    missing = [t for t in tickers if t not in valid]

    if missing:
        st.warning(f"These tickers returned no data for the selected range/interval and were skipped: {', '.join(missing)}")

    if not valid:
        st.error("No valid data to analyze. Try different tickers/date range/interval.")
    else:
        # -------------------------
        # Phase 1: Data Collection (Preview)
        # -------------------------
        st.subheader("📁 Raw Data Preview")
        with st.expander("Show first rows per ticker"):
            for t in valid:
                st.markdown(f"**{t}**")
                st.dataframe(data_dict[t].head())

        # Build Adj Close wide frame and Returns
        adj_close = build_adj_close_frame(data_dict)
        returns = daily_returns(adj_close)

        # Downloads (optional)
        if show_downloads:
            c1, c2 = st.columns(2)
            with c1:
                st.download_button(
                    "⬇️ Download Adjusted Close (CSV)",
                    adj_close.to_csv().encode("utf-8"),
                    file_name="adj_close.csv",
                    mime="text/csv"
                )
            with c2:
                st.download_button(
                    "⬇️ Download Daily Returns (CSV)",
                    returns.to_csv().encode("utf-8"),
                    file_name="daily_returns.csv",
                    mime="text/csv"
                )

        # -------------------------
        # Phase 2: Data Understanding
        # -------------------------
        st.header("🔎 Phase 2 — Data Understanding")

        c1, c2, c3 = st.columns(3)
        with c1:
            st.metric("Tickers analyzed", len(valid))
        with c2:
            st.metric("Date range", f"{adj_close.index.min().date()} → {adj_close.index.max().date()}")
        with c3:
            st.metric("Observations", f"{adj_close.shape[0]:,} rows")

        st.subheader("Summary Statistics — Daily Returns")
        st.dataframe(returns.describe().T)

        st.subheader("Volatility (Std Dev of Daily Returns)")
        st.bar_chart(returns.std().sort_values(ascending=False))

        if len(valid) > 1:
            st.subheader("Correlation Matrix (Daily Returns)")
            fig, ax = plt.subplots(figsize=(6 + 0.3*len(valid), 5 + 0.2*len(valid)))
            sns.heatmap(returns.corr(), annot=True, cmap="coolwarm", ax=ax)
            st.pyplot(fig)

        # Sector comparison if the user has a mix of sectors
        st.subheader("Sector Bucket Comparison (Annualized)")
        # Map each ticker to a simple bucket (first match wins)
        def map_bucket(t: str) -> str:
            for b, syms in SECTOR_BUCKETS.items():
                if t in syms:
                    return b
            return "Other"

        bucket_map = pd.Series({t: map_bucket(t) for t in valid})
        if not returns.empty:
            ann_mean = returns.mean() * 252
            ann_vol  = returns.std() * np.sqrt(252)
            metrics_df = pd.DataFrame({"AnnReturn": ann_mean, "AnnVol": ann_vol, "Bucket": bucket_map})
            st.dataframe(metrics_df.sort_values("AnnVol", ascending=False))
            # Boxplot by bucket of daily returns (stacked long-form)
            long = returns.melt(ignore_index=False, var_name="Ticker", value_name="DailyReturn").dropna()
            long["Bucket"] = long["Ticker"].map(bucket_map)
            fig = px.box(long, x="Bucket", y="DailyReturn", title="Daily Returns by Sector Bucket")
            st.plotly_chart(fig, use_container_width=True)

        # -------------------------
        # Phase 3: Visualization
        # -------------------------
        st.header("📈 Phase 3 — Visualization")

        st.subheader("Normalized Price (Start = 100)")
        norm = normalize_100(adj_close.dropna())
        fig_norm = px.line(norm, x=norm.index, y=norm.columns, title="Normalized Adjusted Close")
        fig_norm.update_xaxes(title="Date")
        fig_norm.update_yaxes(title="Index (100 = start)")
        st.plotly_chart(fig_norm, use_container_width=True)

        st.subheader("Candlestick + Indicators (per Ticker)")
        for t in valid:
            df_t = compute_indicators(data_dict[t], ma_short, ma_long)

            st.markdown(f"**{t}**")
            # Candlestick with BB + SMA overlays
            fig = go.Figure()
            fig.add_trace(go.Candlestick(
                x=df_t.index, open=df_t["Open"], high=df_t["High"], low=df_t["Low"], close=df_t["Close"],
                name="Price"
            ))
            # Overlays only if Adj Close exists to be consistent
            price_series = df_t["Adj Close"] if "Adj Close" in df_t.columns else df_t["Close"]
            fig.add_trace(go.Scatter(x=df_t.index, y=df_t["SMA_short"], mode="lines", name=f"SMA{ma_short}"))
            fig.add_trace(go.Scatter(x=df_t.index, y=df_t["SMA_long"],  mode="lines", name=f"SMA{ma_long}"))
            fig.add_trace(go.Scatter(x=df_t.index, y=df_t["BB_UP"],  mode="lines", name="BB Upper", opacity=0.5))
            fig.add_trace(go.Scatter(x=df_t.index, y=df_t["BB_MID"], mode="lines", name="BB Mid",   opacity=0.5))
            fig.add_trace(go.Scatter(x=df_t.index, y=df_t["BB_LOW"], mode="lines", name="BB Lower", opacity=0.5))
            fig.update_layout(title=f"{t} — Candlestick with SMA & Bollinger Bands", xaxis_rangeslider_visible=False)
            st.plotly_chart(fig, use_container_width=True)

            # RSI
            with st.expander(f"{t} — RSI(14)"):
                fig_rsi = px.line(df_t.reset_index(), x=df_t.index.name or "Date", y="RSI14", title=f"{t} RSI(14)")
                fig_rsi.add_hline(y=70, line_dash="dash")
                fig_rsi.add_hline(y=30, line_dash="dash")
                st.plotly_chart(fig_rsi, use_container_width=True)

            # Volume
            with st.expander(f"{t} — Volume"):
                fig_vol = px.bar(df_t.reset_index(), x=df_t.index.name or "Date", y="Volume", title=f"{t} Volume")
                st.plotly_chart(fig_vol, use_container_width=True)

        # -------------------------
        # Portfolio Insights (Basic)
        # -------------------------
        st.header("💼 Portfolio Snapshot (Equal or Custom Weights)")
        if len(valid) >= 2 and not returns.empty:
            mode = st.radio("Weight Mode", ["Equal Weights", "Custom Weights"], horizontal=True)
            if mode == "Equal Weights":
                weights = pd.Series(1/len(valid), index=valid)
            else:
                st.info("Use sliders to set weights. They must sum to 1.0.")
                sliders = {}
                total_col1, total_col2 = st.columns([3,1])
                with total_col1:
                    cols = st.columns(min(4, len(valid)))
                    current_sum = 0.0
                    for i, t in enumerate(valid):
                        with cols[i % len(cols)]:
                            sliders[t] = st.slider(f"{t}", 0.0, 1.0, 1.0/len(valid), 0.01)
                            current_sum += sliders[t]
                    weights = pd.Series(sliders, dtype=float)
                with total_col2:
                    st.metric("Weight Sum", f"{weights.sum():.2f}")
                if abs(weights.sum() - 1.0) > 1e-6:
                    st.error("Weights must sum to 1.0 to compute portfolio metrics.")
                    st.stop()

            port_ret = to_portfolio(returns[valid], weights)
            metrics = ann_metrics(port_ret)
            c1, c2, c3 = st.columns(3)
            c1.metric("Annualized Return", f"{metrics['ann_return']*100:,.2f}%")
            c2.metric("Annualized Volatility", f"{metrics['ann_vol']*100:,.2f}%")
            c3.metric("Sharpe (no RF)", f"{metrics['sharpe']:.2f}")

            cum_comp = (1 + returns[valid]).cumprod()
            cum_port  = (1 + port_ret).cumprod().rename("Portfolio")
            comp_df = pd.concat([cum_comp, cum_port], axis=1).dropna()

            fig_comp = px.line(comp_df, x=comp_df.index, y=comp_df.columns, title="Cumulative Growth (1 = Start)")
            st.plotly_chart(fig_comp, use_container_width=True)

        # -------------------------
        # Notes
        # -------------------------
        with st.expander("Notes & Tips"):
            st.markdown(
                "- **Intervals**: Some tickers/intervals may produce sparse data (e.g., '1mo'). If charts look empty, try '1d' or expand the date range.\n"
                "- **Adjusted Close** is used for return analysis (dividends/splits). Candlesticks display raw OHLC.\n"
                "- **Caching**: Data is cached to speed up re-runs with the same parameters."
            )

else:
    st.info("Set your tickers/date range and click **Fetch & Analyze** to begin.")
