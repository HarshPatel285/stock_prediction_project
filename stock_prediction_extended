# stock_prediction.py
# Group 9 — Part 1 (kept) + Part 2 (Phases 4 & 5 added)
# Single strong hypothesis, one model family (Random Forest), applied per selected ticker

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from datetime import datetime, date, timedelta

# Viz
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import plotly.express as px

# ML / Stats
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from scipy.stats import ttest_1samp

# New imports for XGBoost and tuning
from xgboost import XGBClassifier
from sklearn.model_selection import RandomizedSearchCV
from scipy.stats import uniform, randint

# -----------------------------
# PAGE CONFIG
# -----------------------------
st.set_page_config(page_title="Case Study • Group 9 • Stocks: EDA + Prediction", layout="wide")

# -----------------------------
# SIDEBAR — INPUTS
# -----------------------------
st.sidebar.title("📊 Case Study — Group 9")

DEFAULT_TICKERS = ["AAPL", "MSFT", "TSLA", "NVDA", "AMZN", "GOOGL", "META", "NFLX"]

tickers = st.sidebar.multiselect("Select tickers", DEFAULT_TICKERS, default=["AAPL", "MSFT", "TSLA", "NVDA"])
min_date = date(2018, 1, 1)
start_date = st.sidebar.date_input("Start date", date.today() - timedelta(days=5 * 365), min_value=min_date)
end_date = st.sidebar.date_input("End date", date.today(), min_value=min_date)
interval = st.sidebar.selectbox("Interval", ["1d", "1wk"])

st.sidebar.markdown("---")
show_rsi = st.sidebar.checkbox("Show RSI (single-ticker view)", True)
show_bb = st.sidebar.checkbox("Show Bollinger Bands (single-ticker view)", True)
sma_fast = st.sidebar.number_input("SMA Fast (days)", 5, 200, 50, 1)
sma_slow = st.sidebar.number_input("SMA Slow (days)", 20, 400, 200, 200)

st.sidebar.markdown("---")
st.sidebar.caption("Phase 4–5 (Model)")
lags = st.sidebar.slider("Lag days (features)", min_value=3, max_value=15, value=5, step=1)
roll_vol = st.sidebar.slider("Rolling volatility window", min_value=3, max_value=60, value=21, step=1)
train_ratio = st.sidebar.slider("Train ratio", min_value=0.5, max_value=0.95, value=0.8, step=0.05)


# -----------------------------
# DATA LOADER
# -----------------------------
@st.cache_data(show_spinner=False)
def fetch_data(tickers, start, end, interval="1d"):
    data = yf.download(tickers, start=start, end=end, interval=interval, auto_adjust=True, threads=True)
    # When multiple tickers, yfinance returns multiindex columns; when single, plain columns.
    panels = {}
    if isinstance(data.columns, pd.MultiIndex):
        for t in tickers:
            df = data.xs(t, axis=1, level=1).copy()
            df = df.dropna()
            df["Return"] = df["Close"].pct_change()
            panels[t] = df
    else:
        # single ticker case
        df = data.copy().dropna()
        df["Return"] = df["Close"].pct_change()
        panels[tickers[0]] = df
    return panels


with st.spinner("Fetching data from Yahoo Finance..."):
    panels = fetch_data(tickers, start_date, end_date, interval=interval)

# -----------------------------
# TITLE
# -----------------------------
st.title("Stock Market Analysis & Prediction — Group 9")
st.caption("University of Niagara Falls Canada • DAMO-611-5 • Phases 1–5")

# -----------------------------
# PHASE 1–3: EDA + VISUALS
# -----------------------------
st.header("Phases 1–3: Data Understanding & Visualization")

# Raw Data (peek for first selected ticker)
if len(tickers) > 0:
    first_t = tickers[0]
    st.subheader(f"Raw Data Preview — {first_t}")
    st.dataframe(panels[first_t].head())

# 1) Normalized Price (start=100)
st.subheader("Normalized Price (start = 100)")
norm_df = pd.DataFrame()
for t in tickers:
    df = panels[t]
    if not df.empty:
        norm = df["Close"] / df["Close"].iloc[0] * 100
        norm_df[t] = norm
norm_df = norm_df.dropna()

if not norm_df.empty:
    fig_norm = px.line(norm_df, title="Normalized Performance")
    st.plotly_chart(fig_norm, use_container_width=True)

# 2) Volatility (std of daily returns)
st.subheader("Volatility (Std Dev of Daily Returns)")
vol = {t: panels[t]["Return"].std() for t in tickers if not panels[t].empty}
if vol:
    vol_df = pd.DataFrame({"Ticker": list(vol.keys()), "Volatility": list(vol.values())}).sort_values("Volatility",
                                                                                                      ascending=False)
    col1, col2 = st.columns([2, 1])
    with col1:
        fig_vol = px.bar(vol_df, x="Ticker", y="Volatility", title="Volatility by Ticker")
        st.plotly_chart(fig_vol, use_container_width=True)
    with col2:
        st.dataframe(vol_df.set_index("Ticker"))

# 3) Correlation Matrix (daily returns)
st.subheader("Correlation Matrix (Daily Returns)")
ret_df = pd.DataFrame({t: panels[t]["Return"] for t in tickers if not panels[t].empty}).dropna()
if not ret_df.empty and ret_df.shape[1] > 1:
    fig_corr = px.imshow(ret_df.corr(), text_auto=True, aspect="auto", title="Correlation Heatmap")
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("Need at least two tickers with overlapping data for correlation.")

# 4) Single-Ticker Deep Dive (Candlestick + SMA + BB + RSI + Volume)
st.subheader("Single-Ticker Technical View")
single = st.selectbox("Choose a ticker to inspect", tickers, index=0 if tickers else None)
if single:
    df = panels[single].copy()
    if df.empty:
        st.warning("No data for selected range.")
    else:
        df["SMA_fast"] = df["Close"].rolling(sma_fast).mean()
        df["SMA_slow"] = df["Close"].rolling(sma_slow).mean()
        if show_bb:
            mid = df["Close"].rolling(20).mean()
            std = df["Close"].rolling(20).std()
            df["BB_mid"] = mid
            df["BB_upper"] = mid + 2 * std
            df["BB_lower"] = mid - 2 * std

        # Candlestick
        candle = go.Figure(data=[
            go.Candlestick(
                x=df.index, open=df["Open"], high=df["High"], low=df["Low"], close=df["Close"], name="Price"
            )
        ])
        candle.update_layout(title=f"{single} — Candlestick", xaxis_rangeslider_visible=False)

        # SMA overlays
        candle.add_trace(go.Scatter(x=df.index, y=df["SMA_fast"], name=f"SMA{sma_fast}", mode="lines"))
        candle.add_trace(go.Scatter(x=df.index, y=df["SMA_slow"], name=f"SMA{sma_slow}", mode="lines"))

        # Bollinger Bands
        if show_bb:
            candle.add_trace(
                go.Scatter(x=df.index, y=df["BB_upper"], name="BB Upper", mode="lines", line=dict(dash="dot")))
            candle.add_trace(go.Scatter(x=df.index, y=df["BB_mid"], name="BB Mid", mode="lines", line=dict(dash="dot")))
            candle.add_trace(
                go.Scatter(x=df.index, y=df["BB_lower"], name="BB Lower", mode="lines", line=dict(dash="dot")))

        st.plotly_chart(candle, use_container_width=True)

        # RSI (14)
        if show_rsi:
            delta = df["Close"].diff()
            gain = delta.clip(lower=0).rolling(14).mean()
            loss = (-delta.clip(upper=0)).rolling(14).mean()
            rs = gain / (loss + 1e-12)
            rsi = 100 - (100 / (1 + rs))
            fig_rsi = go.Figure()
            fig_rsi.add_trace(go.Scatter(x=df.index, y=rsi, mode="lines", name="RSI (14)"))
            fig_rsi.add_hline(y=70, line_dash="dash")
            fig_rsi.add_hline(y=30, line_dash="dash")
            fig_rsi.update_layout(title=f"{single} — RSI (14)")
            st.plotly_chart(fig_rsi, use_container_width=True)

        # Volume
        fig_vol2 = px.bar(df, x=df.index, y="Volume", title=f"{single} — Volume")
        st.plotly_chart(fig_vol2, use_container_width=True)

# 5) Portfolio cumulative growth (equal-weight)
st.subheader("Cumulative Growth — Portfolio vs Constituents (Equal-Weighted)")
cum_df = pd.DataFrame()
for t in tickers:
    df = panels[t]
    if not df.empty:
        cum_df[t] = (1 + df["Return"]).cumprod()
cum_df = cum_df.dropna()
if not cum_df.empty:
    port = cum_df.mean(axis=1)
    fig_cum = go.Figure()
    for t in cum_df.columns:
        fig_cum.add_trace(go.Scatter(x=cum_df.index, y=cum_df[t], mode="lines", name=t))
    fig_cum.add_trace(go.Scatter(x=port.index, y=port, mode="lines", name="Equal-Weighted Portfolio",
                                 line=dict(width=3, dash="dash")))
    fig_cum.update_layout(title="Cumulative Growth")
    st.plotly_chart(fig_cum, use_container_width=True)

# -----------------------------
# PHASES 4–5: MODEL BUILDING & EVALUATION
# -----------------------------
st.header("Phases 4–5: Predictive Modeling & Evaluation")

st.markdown(
    """
    **Hypothesis:**
    **H0:** Next-day direction of returns is not predictable better than a 50% baseline (random walk).
    **H1:** A XGBoost classifier with hyperparameter tuning predicts next-day direction significantly better than 50%.
    """
)


def build_features(df: pd.DataFrame, lags: int, vol_window: int) -> pd.DataFrame:
    X = df.copy()
    X["Ret"] = X["Close"].pct_change()

    # Create lagged features for returns
    for i in range(1, lags + 1):
        X[f"Ret_lag_{i}"] = X["Ret"].shift(i)

    # Add volatility
    X["Vol_roll"] = X["Ret"].rolling(vol_window).std()

    # Add technical indicators
    # RSI (14-day)
    delta = X["Close"].diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rs = gain / (loss + 1e-12)
    X["RSI"] = 100 - (100 / (1 + rs))

    # MACD (12, 26, 9)
    exp12 = X["Close"].ewm(span=12, adjust=False).mean()
    exp26 = X["Close"].ewm(span=26, adjust=False).mean()
    X["MACD"] = exp12 - exp26
    X["MACD_Signal"] = X["MACD"].ewm(span=9, adjust=False).mean()
    X["MACD_Hist"] = X["MACD"] - X["MACD_Signal"]
    X["MACD_Diff"] = X["MACD"].diff()

    # Stochastic Oscillator
    low14 = X["Low"].rolling(14).min()
    high14 = X["High"].rolling(14).max()
    X["Stochastic_K"] = 100 * ((X["Close"] - low14) / (high14 - low14))
    X["Stochastic_D"] = X["Stochastic_K"].rolling(3).mean()
    X["Stochastic_Diff"] = X["Stochastic_K"].diff()

    # Classification target: up (1) if next-day return > 0, else 0
    X["Target"] = (X["Ret"].shift(-1) > 0).astype(int)
    X = X.dropna()

    feature_cols = [
                       c for c in X.columns if c.startswith("Ret_lag_")
                   ] + ["Vol_roll", "RSI", "MACD", "MACD_Signal", "MACD_Hist", "MACD_Diff", "Stochastic_K",
                        "Stochastic_D", "Stochastic_Diff"]

    return X, feature_cols


def train_eval_classifier(df: pd.DataFrame, lags: int, vol_window: int, train_ratio: float):
    X_full, feats = build_features(df, lags, vol_window)
    if X_full.empty or len(X_full) < 200:
        return None  # not enough data
    n = len(X_full)
    split = int(n * train_ratio)
    Xtrain, Xtest = X_full.iloc[:split], X_full.iloc[split:]
    Xtr, ytr = Xtrain[feats], Xtrain["Target"]
    Xte, yte = Xtest[feats], Xtest["Target"]

    # ---- NEW: XGBoost with Hyperparameter Tuning ----
    st.write("Optimizing XGBoost model with Randomized Search...")
    param_dist = {
        'n_estimators': randint(50, 400),
        'max_depth': randint(1, 10),
        'learning_rate': uniform(0.01, 0.3),
        'subsample': uniform(0.5, 0.4),
        'colsample_bytree': uniform(0.5, 0.4)
    }
    xgb_model = XGBClassifier(use_label_encoder=False, eval_metric='logloss', random_state=42)
    random_search = RandomizedSearchCV(
        xgb_model,
        param_distributions=param_dist,
        n_iter=20,  # Number of parameter settings that are sampled
        cv=3,  # Number of cross-validation folds
        scoring='accuracy',
        n_jobs=-1,
        random_state=42
    )

    random_search.fit(Xtr, ytr)
    best_model = random_search.best_estimator_
    pred = best_model.predict(Xte)
    # ---- END NEW ----

    acc = accuracy_score(yte, pred)
    prec = precision_score(yte, pred, zero_division=0)
    rec = recall_score(yte, pred, zero_division=0)
    f1 = f1_score(yte, pred, zero_division=0)
    cm = confusion_matrix(yte, pred)

    # t-test vs 0.5 (random): correctness vector ~ Bernoulli
    correct = (pred == yte).astype(int).values
    t_stat, p_val = ttest_1samp(correct, 0.5)

    # Strategy backtest: long when model predicts UP, else cash
    strat = Xtest.copy()
    strat["Pred"] = pred
    strat["NextRet"] = strat["Ret"].shift(
        -0)  # already aligned to predict next day direction; here use realized return at t (following day from training perspective)
    # execute decision at t-1 to capture next day's ret → shift Pred by 1 to avoid lookahead
    strat["Signal"] = strat["Pred"].shift(1).fillna(0)
    strat["StratRet"] = strat["Signal"] * strat["Ret"]
    strat_cum = (1 + strat["StratRet"].fillna(0)).cumprod()
    bh_cum = (1 + strat["Ret"]).cumprod()

    return {
        "model": best_model,
        "features": feats,
        "metrics": {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1},
        "cm": cm,
        "p_value": float(p_val),
        "t_stat": float(t_stat),
        "cum_strat": strat_cum,
        "cum_bh": bh_cum,
        "test_index": strat.index,
        "best_params": random_search.best_params_  # Return best parameters for reporting
    }


# Run model per selected ticker
results = {}
for t in tickers:
    df = panels[t]
    if df.empty:
        continue
    res = train_eval_classifier(df, lags=lags, vol_window=roll_vol, train_ratio=train_ratio)
    if res:
        results[t] = res

if not results:
    st.warning("Not enough overlapping data to train models for the selected settings.")
else:
    # Summary cards
    st.subheader("Per-Ticker Results")
    cols = st.columns(min(4, len(results)))
    i = 0
    for t, res in results.items():
        m = res["metrics"]
        with cols[i % len(cols)]:
            st.metric(f"{t} • Accuracy", f"{m['accuracy']:.2%}")
            st.caption(f"F1: {m['f1']:.2f} • p={res['p_value']:.4f}")
        i += 1

    # Detailed tabs per ticker
    tab_titles = [f"{t}" for t in results.keys()]
    tabs = st.tabs(tab_titles)
    for tab, (t, res) in zip(tabs, results.items()):
        with tab:
            st.markdown(f"### {t}: Metrics & Tests")
            st.markdown(f"**Best Parameters:** `{res['best_params']}`")
            m = res["metrics"]
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy", f"{m['accuracy']:.2%}")
            c2.metric("Precision", f"{m['precision']:.2f}")
            c3.metric("Recall", f"{m['recall']:.2f}")
            c4.metric("F1", f"{m['f1']:.2f}")

            st.write(f"**t-stat:** {res['t_stat']:.2f} • **p-value:** {res['p_value']:.4f}")
            if res["p_value"] < 0.05:
                st.success("Reject H0: Model accuracy is significantly above 50% at α = 0.05.")
            else:
                st.warning("Fail to reject H0: No significant improvement over 50% baseline.")

            # Confusion Matrix
            cm = res["cm"]
            fig_cm = px.imshow(cm, text_auto=True, aspect="auto", color_continuous_scale="Blues")
            fig_cm.update_layout(title="Confusion Matrix", xaxis_title="Predicted", yaxis_title="Actual")
            fig_cm.update_xaxes(ticktext=["Down", "Up"], tickvals=[0, 1])
            fig_cm.update_yaxes(ticktext=["Down", "Up"], tickvals=[0, 1])
            st.plotly_chart(fig_cm, use_container_width=True)

            # Strategy vs Buy & Hold
            st.markdown("#### Strategy Backtest: Model (long if predict Up) vs Buy & Hold")
            idx = res["test_index"]
            strat = res["cum_strat"].reindex(idx)
            bh = res["cum_bh"].reindex(idx)
            fig_bt = go.Figure()
            fig_bt.add_trace(go.Scatter(x=idx, y=strat, name="Model Strategy", mode="lines"))
            fig_bt.add_trace(go.Scatter(x=idx, y=bh, name="Buy & Hold", mode="lines"))
            fig_bt.update_layout(title=f"{t} — Cumulative Returns (Test Set)")
            st.plotly_chart(fig_bt, use_container_width=True)

    # Portfolio-level strategy (aggregate across tickers, equal weight)
    st.subheader("Aggregate Strategy (Across Selected Tickers, Equal Weight)")
    # Align on common test index
    common_index = None
    for r in results.values():
        common_index = r["test_index"] if common_index is None else common_index.intersection(r["test_index"])
    if common_index is not None and len(common_index) > 10:
        strat_mat = []
        bh_mat = []
        for r in results.values():
            s = r["cum_strat"].reindex(common_index).pct_change().fillna(0)
            b = r["cum_bh"].reindex(common_index).pct_change().fillna(0)
            strat_mat.append(s.values)
            bh_mat.append(b.values)
        # equal weight average of daily returns
        strat_port = (1 + np.mean(np.vstack(strat_mat), axis=0)).cumprod()
        bh_port = (1 + np.mean(np.vstack(bh_mat), axis=0)).cumprod()
        fig_port = go.Figure()
        fig_port.add_trace(go.Scatter(x=common_index, y=strat_port, name="Model Strategy (EW)", mode="lines"))
        fig_port.add_trace(go.Scatter(x=common_index, y=bh_port, name="Buy & Hold (EW)", mode="lines"))
        fig_port.update_layout(title="Portfolio-Level Cumulative Returns (Test Period, Equal Weight)")
        st.plotly_chart(fig_port, use_container_width=True)
    else:
        st.info("Not enough overlapping test dates to compute an aggregate strategy across all selected tickers.")

    # Business Interpretation Box
    st.subheader("📌 Business Interpretation")
    # Aggregate stats for narrative
    accs = [res["metrics"]["accuracy"] for res in results.values()]
    sig = [res for res in results.values() if res["p_value"] < 0.05]
    if results:
        top = max(results.items(), key=lambda kv: kv[1]["metrics"]["accuracy"])
        st.markdown(
            f"""
- **Overall mean accuracy:** **{np.mean(accs):.2%}** across {len(accs)} tickers; **{len(sig)}** of {len(results)} tickers show accuracy **significantly > 50%** at α=0.05.
- **Best performing ticker:** **{top[0]}** with accuracy **{top[1]['metrics']['accuracy']:.2%}**.
- **Investor takeaway:**
  - Where significance holds, a simple **directional timing overlay** (long if predicted up, else cash) can alter risk/return versus buy-and-hold.
  - **Stable anchors (e.g., MSFT/AAPL)** typically show steadier signals; **high-vol names (e.g., TSLA/NVDA)** can amplify gains *and* drawdowns.
  - Use model signals as a **satellite overlay** on a diversified core, and prefer **portfolio-level aggregation** of signals to reduce idiosyncratic noise.
"""
        )
    else:
        st.warning("No model results available for business interpretation.")

# -----------------------------
# FOOTER — METHODS (for grading clarity)
# -----------------------------
with st.expander("🧪 Methods & Reproducibility (Phases 4–5)"):
    st.markdown(
        f"""
- **Features:** past **{lags}** lagged daily returns, rolling volatility (window={roll_vol}), and **technical indicators (RSI, MACD, Stochastic)**.
- **Model family:** **XGBoost Classifier** with **Randomized Search** for hyperparameter tuning.
- **Target:** next-day direction (Up=1 if return > 0 else 0).
- **Split:** chronological (train={int(train_ratio * 100)}% / test={int((1 - train_ratio) * 100)}%).
- **Metrics:** Accuracy, Precision, Recall, F1, Confusion Matrix.
- **Statistical test:** one-sample **t-test** on correctness vs. 0.5 (random), per ticker.
- **Backtest:** Long when model predicts Up (signal shifted to avoid look-ahead), else cash; compared to buy-and-hold on the same test window.
- **Hypothesis decision rule:** Reject H0 if p < 0.05.
"""
    )
