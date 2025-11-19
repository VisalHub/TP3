# Implement the function to make the tests pass.
# •	All normalized values are within [0, 1].
# •	Output column length matches input.
# •	Invalid column name raises a KeyError.
import pandas as pd

def normalize_column(df, column):
    if column not in df.columns:
        raise KeyError(f"Column '{column}' does not exist in DataFrame.")
    
    col_min = df[column].min()
    col_max = df[column].max()
    
    if col_max - col_min == 0:
        df[column] = 0.0
    else:
        df[column] = (df[column] - col_min) / (col_max - col_min)
    
    return df

