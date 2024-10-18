import pandas as pd

def from_file(filename, delimiter):
    # Load the data into a DataFrame
    df = pd.read_csv(filename, delimiter=delimiter)
    
    return df