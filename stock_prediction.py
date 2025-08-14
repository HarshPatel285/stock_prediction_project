# streamlit_app.py
import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

st.set_page_config(page_title="Financial Data Analytics", layout="wide")

# ----------------------
# Title
# ----------------------
st.title("📊 Financial Data Analytics Dashboard")
st.markdown("This app covers **Phase 1–3**: Data Collection, Understanding, and Visualization.")

# ----------------------
# Sidebar inputs
# ----------------------
st.sidebar.header("Settings")
tickers = st.sidebar.multiselect(
    "Select Stocks",
    ["AAPL", "MSFT", "TSLA", "JPM", "GS", "RY", "AMZN", "GOOGL"],
    default=["AAPL", "MSFT", "TSLA", "JPM"]
)

start_date = st.sidebar.date_input("Start Date", datetime(2019, 1, 1))
end_date = st.sidebar.date_input("End Date", datetime(2024, 12, 31))

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

        st.subheader("📁 Raw Data Preview")
        st.write(data.head())

        # ------------------------------------------------------
        # Phase 2: Data Understanding
        # ------------------------------------------------------
        st.header("🔎 Phase 2: Data Understanding")

        if isinstance(data.columns, pd.MultiIndex):
            # Multi-stock case
            close_data = pd.concat([data[t]['Adj Close'] for t in tickers], axis=1)
            close_data.columns = tickers
        else:
            # Single stock case
            close_data = data['Adj Close'].to_frame(name=tickers[0])

        # Daily Returns
        returns = close_data.pct_change().dropna()

        # Summary Statistics
        st.subheader("📊 Summary Statistics (Returns)")
        st.write(returns.describe())

        # Volatility comparison
        vol = returns.std().sort_values(ascending=False)
        st.subheader("📉 Stock Volatility (Standard Deviation of Returns)")
        st.bar_chart(vol)

        # Correlation
        corr_matrix = returns.corr()
        st.subheader("🔗 Correlation Matrix of Returns")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.heatmap(corr_matrix, annot=True, cmap="coolwarm", ax=ax)
        st.pyplot(fig)

        # ------------------------------------------------------
        # Phase 3: Data Visualization
        # ------------------------------------------------------
        st.header("📈 Phase 3: Data Visualization")

        # Price Trends
        st.subheader("📈 Price Trends")
        st.line_chart(close_data)

        # Distribution of Returns
        st.subheader("📊 Distribution of Daily Returns")
        fig, ax = plt.subplots(figsize=(10, 6))
        for ticker in tickers:
            sns.histplot(returns[ticker], bins=50, kde=True, label=ticker, ax=ax, alpha=0.5)
        plt.legend()
        st.pyplot(fig)

        # Boxplot of Volatility
        st.subheader("📦 Volatility Comparison (Boxplot of Returns)")
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.boxplot(data=returns, ax=ax)
        st.pyplot(fig)

        # Option to Download Cleaned Data
        st.subheader("💾 Download Processed Data")
        csv = close_data.to_csv().encode("utf-8")
        st.download_button(
            label="Download Adjusted Close Prices CSV",
            data=csv,
            file_name="cleaned_financial_data.csv",
            mime="text/csv",
        )