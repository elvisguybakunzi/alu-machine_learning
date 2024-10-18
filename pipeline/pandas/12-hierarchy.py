#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

df1 = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')
df2 = from_file('bitstampUSD_1-min_data_2012-01-01_to_2020-04-22.csv', ',')

# Set 'Timestamp' as index for both DataFrames
df1 = df1.set_index('Timestamp')
df2 = df2.set_index('Timestamp')

# Select data from timestamps 1417411980 to 1417417980, inclusive
df1_subset = df1.loc[1417411980:1417417980]
df2_subset = df2.loc[1417411980:1417417980]

# Concatenate the DataFrames
df = pd.concat([df2_subset, df1_subset], keys=['bitstamp', 'coinbase'], axis=1)

# Rearrange the MultiIndex levels
df = df.swaplevel(0, 1, axis=1)

# Sort the DataFrame by timestamp
df = df.sort_index()

print(df)