import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta

# ---------------------------------------------
# Sidebar: User Input
# ---------------------------------------------
st.sidebar.header("Stock Dashboard Settings")

# Select stock ticker
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, MSFT, TSLA)", "AAPL")

# Date range selection
end_date = st.sidebar.date_input("End Date", datetime.today())
start_date = st.sidebar.date_input("Start Date", end_date - timedelta(days=365))

# Interval
interval = st.sidebar.selectbox("Interval", ["1d", "1wk", "1mo"])

# ---------------------------------------------
# Data Collection
# ---------------------------------------------
@st.cache_data
def load_data(ticker, start, end, interval):
    df = yf.download(ticker, start=start, end=end, interval=interval)
    df.reset_index(inplace=True)
    return df

data = load_data(ticker, start_date, end_date, interval)

st.title(f"📊 Stock Market Dashboard: {ticker}")
st.write(f"Showing data for **{ticker}** from {start_date} to {end_date}")

# ---------------------------------------------
# Data Understanding
# ---------------------------------------------
st.subheader("Data Overview")
st.write(data.head())

st.subheader("Statistical Summary")
st.write(data.describe())

# Show raw data option
if st.checkbox("Show Raw Data"):
    st.write(data)

# ---------------------------------------------
# Data Visualization
# ---------------------------------------------
st.subheader("Stock Price Visualization")

# Candlestick Chart
fig_candle = go.Figure(data=[go.Candlestick(
    x=data['Date'],
    open=data['Open'],
    high=data['High'],
    low=data['Low'],
    close=data['Close'],
    name='Candlestick'
)])
fig_candle.update_layout(title=f"{ticker} Candlestick Chart", xaxis_rangeslider_visible=True)
st.plotly_chart(fig_candle, use_container_width=True)

# Line Chart with Options
chart_type = st.selectbox("Select Line Chart Type", ["Close", "Open", "High", "Low", "Adj Close"])
fig_line = px.line(data, x="Date", y=chart_type, title=f"{ticker} {chart_type} Price")
st.plotly_chart(fig_line, use_container_width=True)

# Volume Chart
fig_vol = px.bar(data, x="Date", y="Volume", title=f"{ticker} Trading Volume")
st.plotly_chart(fig_vol, use_container_width=True)

# Moving Average
st.subheader("Moving Averages")
ma_window = st.slider("Select Moving Average Window (days)", 5, 50, 20)
data[f"MA_{ma_window}"] = data['Close'].rolling(ma_window).mean()
fig_ma = px.line(data, x="Date", y=["Close", f"MA_{ma_window}"], title=f"{ticker} Closing Price with {ma_window}-Day MA")
st.plotly_chart(fig_ma, use_container_width=True)

# Correlation Heatmap (optional if multiple stocks later)
if st.checkbox("Show Correlation Heatmap"):
    corr = data.corr(numeric_only=True)
    fig_corr = px.imshow(corr, text_auto=True, title="Correlation Heatmap")
    st.plotly_chart(fig_corr, use_container_width=True)
