#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

# Load the DataFrame
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Slice the DataFrame along the columns 'High', 'Low', 'Close', and 'Volume_(BTC)' and take every 60th row
df = df[['High', 'Low', 'Close', 'Volume_(BTC)']].iloc[::60]

print(df.tail())
