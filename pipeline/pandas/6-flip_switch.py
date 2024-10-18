#!/usr/bin/env python3

import pandas as pd
from_file = __import__('2-from_file').from_file

# Load the DataFrame
df = from_file('coinbaseUSD_1-min_data_2014-12-01_to_2019-01-09.csv', ',')

# Sort the DataFrame in reverse chronological order by 'Timestamp'
df = df.sort_values('Timestamp', ascending=False)

# Transpose the DataFrame (rows become columns and vice versa)
df = df.transpose()

print(df.tail(8))
