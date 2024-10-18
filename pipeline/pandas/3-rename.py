#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

# Load the DataFrame
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Rename the 'Timestamp' column to 'Datetime'
df.rename(columns={'Timestamp': 'Datetime'}, inplace=True)

# Convert the 'Datetime' column from timestamp to datetime format
df['Datetime'] = pd.to_datetime(df['Datetime'], unit='s')

# Select and display only the 'Datetime' and 'Close' columns
df = df[['Datetime', 'Close']]

print(df.tail())
