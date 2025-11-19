# Write tests for a clean_data(df) function that removes duplicates and nulls.
# •	Duplicates are removed correctly.
# •	All null values are dropped.
# •	The number of rows decreases after cleaning when nulls or duplicates exist.

import pandas as pd 

def clean_data(df):
    df = df.drop_duplicates()
    df = df.dropna()
    return df
