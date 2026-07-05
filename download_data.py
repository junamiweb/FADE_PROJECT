import pandas as pd
import yfinance as yf

print("Downloading BTC data...")

# BTC-USD data (1h interval)
df = yf.download(
    tickers="BTC-USD",
    interval="1h",
    period="730d"  # ~2 years
)

# yfinance can return MultiIndex columns (field, ticker); flatten to field only
if isinstance(df.columns, pd.MultiIndex):
    df.columns = df.columns.get_level_values(0)

df = df.reset_index()

# normalize column names for FADE
df = df.rename(columns={
    "Datetime": "timestamp",
    "Date": "timestamp",
    "Open": "open",
    "High": "high",
    "Low": "low",
    "Close": "close",
    "Volume": "volume"
})

# keep only needed columns
df = df[["timestamp", "open", "high", "low", "close", "volume"]]

df.to_csv("btc.csv", index=False)

print("Saved as btc.csv")
print(df.tail())