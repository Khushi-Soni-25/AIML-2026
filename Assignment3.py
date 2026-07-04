"""
Assignment 3: Encode categorical variables using LabelEncoder & OneHotEncoder
Dataset: Titanic-Dataset.csv
Author: Khushi Soni
"""

import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

file_path = "Titanic-Dataset.csv"
try:
    df = pd.read_csv(file_path)
    print(f"Successfully loaded '{file_path}'\n")
except FileNotFoundError:
    print(f"ERROR: File '{file_path}' not found.")
    exit()

print("ORIGINAL DATASET - FIRST 5 ROWS")
print(df.head())

print("STEP 1: HANDLING MISSING VALUES")
print("Missing values BEFORE imputation:")
print(df.isnull().sum())
print("\n")

if df['Age'].isnull().sum() > 0:
    age_median = df['Age'].median()
    df['Age'].fillna(age_median, inplace=True)
    print(f"Imputed 'Age' with MEDIAN = {age_median:.2f}")

if df['Fare'].isnull().sum() > 0:
    fare_mean = df['Fare'].mean()
    df['Fare'].fillna(fare_mean, inplace=True)
    print(f"Imputed 'Fare' with MEAN = {fare_mean:.2f}")

if df['Embarked'].isnull().sum() > 0:
    embarked_mode = df['Embarked'].mode()[0]
    df['Embarked'].fillna(embarked_mode, inplace=True)
    print(f"Imputed 'Embarked' with MODE = '{embarked_mode}'")

if df['Cabin'].isnull().sum() > 0:
    df['Cabin'].fillna('Unknown', inplace=True)
    print(f"Filled 'Cabin' missing with 'Unknown'")

print("\nMissing values AFTER imputation:")
print(df.isnull().sum())


print("STEP 2: ENCODING CATEGORICAL VARIABLES")
print("\n--- Using LabelEncoder for 'Sex' ---")
le = LabelEncoder()
df['Sex_encoded'] = le.fit_transform(df['Sex'])
print(f"Mapping: {dict(zip(le.classes_, le.transform(le.classes_)))}")
# male -> 1, female -> 0 (by default)

print("\nFirst 5 rows of 'Sex' vs 'Sex_encoded':")
print(df[['Sex', 'Sex_encoded']].head())

print("\n--- Using OneHotEncoder for 'Embarked' ---")
ohe = OneHotEncoder(sparse_output=False, drop='first')  # drop='first' avoids dummy variable trap
embarked_encoded = ohe.fit_transform(df[['Embarked']])

feature_names = ohe.get_feature_names_out(['Embarked'])
embarked_df = pd.DataFrame(embarked_encoded, columns=feature_names, index=df.index)

print(f"OneHotEncoder created {len(feature_names)} new columns: {list(feature_names)}")
print("\nFirst 5 rows of OneHotEncoded columns:")
print(embarked_df.head())

df_final = df.drop(columns=['Sex', 'Embarked'])
df_final = pd.concat([df_final, embarked_df], axis=1)

print("FINAL DATASET AFTER ENCODING - FIRST 5 ROWS")
print(df_final.head())

output_file = "Titanic_cleaned_encoded.csv"
df_final.to_csv(output_file, index=False)
print(f"\nFully cleaned & encoded dataset saved as '{output_file}'")

print("FINAL SUMMARY")
print(f"Total rows: {df_final.shape[0]}")
print(f"Total columns: {df_final.shape[1]}")
print(f"Column names: {list(df_final.columns)}")
print("\nAssignment 3 complete! Ready for Mini Project.")
