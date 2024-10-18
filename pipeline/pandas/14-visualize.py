#!/usr/bin/env python3

from datetime import date
import matplotlib.pyplot as plt
import pandas as pd
from_file = __import__('2-from_file').from_file

df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Remove 'Weighted_Price' column
df = df.drop('Weighted_Price', axis=1)

# Rename 'Timestamp' to 'Date' and convert to datetime
df['Date'] = pd.to_datetime(df['Timestamp'], unit='s')
df = df.drop('Timestamp', axis=1)

# Set 'Date' as index
df = df.set_index('Date')

# Fill missing values
df['Close'] = df['Close'].fillna(method='ffill')
for col in ['High', 'Low', 'Open']:
    df[col] = df[col].fillna(df['Close'])
df['Volume_(BTC)'] = df['Volume_(BTC)'].fillna(0)
df['Volume_(Currency)'] = df['Volume_(Currency)'].fillna(0)

# Filter data from 2017 onwards and resample to daily intervals
df_2017 = df['2017':]
df_daily = df_2017.resample('D').agg({
    'High': 'max',
    'Low': 'min',
    'Open': 'mean',
    'Close': 'mean',
    'Volume_(BTC)': 'sum',
    'Volume_(Currency)': 'sum'
})

# Plotting
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 16), sharex=True)

# Plot prices
ax1.plot(df_daily.index, df_daily['High'], label='High', alpha=0.8)
ax1.plot(df_daily.index, df_daily['Low'], label='Low', alpha=0.8)
ax1.plot(df_daily.index, df_daily['Open'], label='Open', alpha=0.8)
ax1.plot(df_daily.index, df_daily['Close'], label='Close', alpha=0.8)
ax1.set_ylabel('Price (USD)')
ax1.legend()
ax1.set_title('Bitcoin Prices (2017-2019)')

# Plot volume
ax2.bar(df_daily.index, df_daily['Volume_(BTC)'], label='Volume (BTC)', alpha=0.8)
ax2.set_ylabel('Volume (BTC)')
ax2.legend()
ax2.set_title('Bitcoin Volume (2017-2019)')

plt.xlabel('Date')
plt.tight_layout()
plt.show()