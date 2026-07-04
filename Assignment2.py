"""
Assignment 2: Handle missing data using mean/median imputation
Dataset: Titanic-Dataset.csv
Author: Khushi Soni
"""

import pandas as pd
import numpy as np

file_path = "Titanic-Dataset.csv"

try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded '{file_path}'\n")
except FileNotFoundError:
    print(f"ERROR: File '{file_path}' not found.")
    exit()

print("BEFORE IMPUTATION - MISSING VALUES COUNT")
print(df.isnull().sum())

numeric_cols = df.select_dtypes(include=[np.number]).columns
print(f"Numerical columns: {list(numeric_cols)}")

null_in_numeric = df[numeric_cols].isnull().sum()
null_in_numeric = null_in_numeric[null_in_numeric > 0]

if len(null_in_numeric) == 0:
    print("No missing values in numerical columns.")
else:
    print(f" Numerical columns with missing values:\n{null_in_numeric}")

if df['Age'].isnull().sum() > 0:
    age_median = df['Age'].median()
    df['Age'].fillna(age_median, inplace=True)
    print(f"Imputed 'Age' with MEDIAN = {age_median:.2f}")

if df['Fare'].isnull().sum() > 0:
    fare_mean = df['Fare'].mean()
    df['Fare'].fillna(fare_mean, inplace=True)
    print(f"Imputed 'Fare' with MEAN = {fare_mean:.2f}")

for col in numeric_cols:
    if df[col].isnull().sum() > 0 and col not in ['Age', 'Fare']:
        median_val = df[col].median()
        df[col].fillna(median_val, inplace=True)
        print(f"Imputed '{col}' with MEDIAN = {median_val:.2f}")

if df['Embarked'].isnull().sum() > 0:
    embarked_mode = df['Embarked'].mode()[0]
    df['Embarked'].fillna(embarked_mode, inplace=True)
    print(f"Imputed 'Embarked' with MODE = '{embarked_mode}'")

if df['Cabin'].isnull().sum() > 0:
    df['Cabin'].fillna('Unknown', inplace=True)
    print(f"Filled 'Cabin' missing with 'Unknown'")

print("AFTER IMPUTATION - MISSING VALUES COUNT")
print(df.isnull().sum())

print("SUMMARY")
print(f"Total rows: {df.shape[0]}")
print(f"Total columns: {df.shape[1]}")
print(f"Remaining missing values: {df.isnull().sum().sum()} (should be 0 if all handled)")

print("\nFirst 10 rows of 'Age' after imputation:")
print(df['Age'].head(10))
