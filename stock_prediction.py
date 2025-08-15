# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objs as go
from datetime import datetime

st.set_page_config(page_title="Financial Data Analytics", layout="wide")

# ----------------------
# Title
# ----------------------
st.title("📊 Advanced Financial Analytics Dashboard")
st.markdown("Covers **Phase 1–3** with advanced options: Technical Indicators, Multiple Charts, and Portfolio Insights.")

# ----------------------
# Sidebar Inputs
# ----------------------
st.sidebar.header("Settings")
tickers = st.sidebar.multiselect(
    "Select Stocks",
    ["AAPL", "MSFT", "TSLA", "JPM", "GS", "RY", "AMZN", "GOOGL"],
    default=["AAPL", "MSFT", "TSLA"]
)

start_date = st.sidebar.date_input("Start Date", datetime(2019, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime(2024, 12, 31))

chart_type = st.sidebar.selectbox(
    "Choose Chart Type",
    ["Line Chart", "Candlestick Chart", "Moving Averages"]
)

# ----------------------
# Fetch Data
# ----------------------
if st.sidebar.button("Fetch & Analyze Data"):
    if len(tickers) == 0:
        st.warning("⚠️ Please select at least one stock.")
    else:
        st.success(f"Fetching data for: {', '.join(tickers)}")

        data = yf.download(
            tickers,
            start=start_date,
            end=end_date,
            interval="1d",
            group_by="ticker",
            auto_adjust=False
        )

        # Handle multi/single stock case
        if isinstance(data.columns, pd.MultiIndex):
            close_data = pd.concat([data[t]['Adj Close'] for t in tickers], axis=1)
            close_data.columns = tickers
        else:
            close_data = data['Adj Close'].to_frame(name=tickers[0])

        returns = close_data.pct_change().dropna()

        # ------------------------------------------------------
        # Phase 2: Data Understanding
        # ------------------------------------------------------
        st.header("🔎 Phase 2: Data Understanding")
        st.subheader("📊 Summary Statistics (Returns)")
        st.write(returns.describe())

        st.subheader("📉 Volatility (Std Dev of Returns)")
        st.bar_chart(returns.std().sort_values(ascending=False))

        st.subheader("🔗 Correlation Matrix")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(returns.corr(), annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # ------------------------------------------------------
        # Phase 3: Visualization Options
        # ------------------------------------------------------
        st.header("📈 Phase 3: Visualization")

        for ticker in tickers:
            st.subheader(f"📊 {ticker} - {chart_type}")

            df = data[ticker] if isinstance(data.columns, pd.MultiIndex) else data

            if chart_type == "Line Chart":
                st.line_chart(df["Adj Close"])

            elif chart_type == "Candlestick Chart":
                fig = go.Figure(data=[go.Candlestick(
                    x=df.index,
                    open=df["Open"], high=df["High"],
                    low=df["Low"], close=df["Close"]
                )])
                fig.update_layout(title=f"Candlestick - {ticker}", xaxis_rangeslider_visible=False)
                st.plotly_chart(fig, use_container_width=True)

            elif chart_type == "Moving Averages":
                df["MA50"] = df["Adj Close"].rolling(window=50).mean()
                df["MA200"] = df["Adj Close"].rolling(window=200).mean()

                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(df.index, df["Adj Close"], label="Adj Close", alpha=0.7)
                ax.plot(df.index, df["MA50"], label="MA50", linestyle="--")
                ax.plot(df.index, df["MA200"], label="MA200", linestyle="--")
                ax.legend()
                ax.set_title(f"{ticker} with Moving Averages")
                st.pyplot(fig)

        # ------------------------------------------------------
        # Extra: Technical Indicators
        # ------------------------------------------------------
        st.header("⚡ Technical Indicators")

        for ticker in tickers:
            df = data[ticker] if isinstance(data.columns, pd.MultiIndex) else data
            df["Returns"] = df["Adj Close"].pct_change()

            # RSI Calculation
            delta = df["Adj Close"].diff()
            gain = (delta.where(delta > 0, 0)).rolling(14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
            rs = gain / loss
            df["RSI"] = 100 - (100 / (1 + rs))

            st.subheader(f"{ticker} - RSI (Relative Strength Index)")
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.plot(df.index, df["RSI"], label="RSI", color="purple")
            ax.axhline(70, color="red", linestyle="--")
            ax.axhline(30, color="green", linestyle="--")
            ax.set_title(f"{ticker} RSI")
            st.pyplot(fig)

        # ------------------------------------------------------
        # Portfolio Analysis (Bonus)
        # ------------------------------------------------------
        st.header("💼 Portfolio Insights")

        st.markdown("Compare mean return vs risk (volatility) of selected stocks.")
        mean_returns = returns.mean() * 252  # Annualized
        volatilities = returns.std() * np.sqrt(252)

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.scatter(volatilities, mean_returns, c="blue", s=100)
        for i, txt in enumerate(tickers):
            ax.annotate(txt, (volatilities[i], mean_returns[i]))
        ax.set_xlabel("Volatility (Risk)")
        ax.set_ylabel("Expected Return")
        ax.set_title("Risk vs Return")
        st.pyplot(fig)

        # ------------------------------------------------------
        # Download Data
        # ------------------------------------------------------
        st.subheader("💾 Download Processed Data")
        csv = close_data.to_csv().encode("utf-8")
        st.download_button(
            label="Download Adjusted Close Prices CSV",
            data=csv,
            file_name="enhanced_financial_data.csv",
            mime="text/csv",
        )
