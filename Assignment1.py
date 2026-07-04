"""
Assignment 1: Load a dataset using Pandas and summarize basic stats
Dataset: Titanic-Dataset.csv (provided locally)
Author: Khushi Soni
"""

import pandas as pd
import numpy as np

file_path = "Titanic-Dataset.csv"

try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded '{file_path}'")
except FileNotFoundError:
    print(f"ERROR: File '{file_path}' not found. Please check the path.")
    exit()  

print("FIRST 5 ROWS OF THE DATASET")
print(df.head())

print("DATASET INFO (.info())")
print(df.info())   

print("DESCRIPTIVE STATISTICS (NUMERIC COLUMNS) - .describe()")
print(df.describe())

print("DESCRIPTIVE STATISTICS (ALL COLUMNS) - .describe(include='all')")
print(df.describe(include='all'))


print("ADDITIONAL DATASET INFO")
print(f"Total rows: {df.shape[0]}")
print(f"Total columns: {df.shape[1]}")
print(f"Column names: {list(df.columns)}")
print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.2f} KB")
