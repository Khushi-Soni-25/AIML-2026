"""
Mini Project 1: Titanic Survival Prediction – Data Cleaning Project
Dataset: Titanic-Dataset.csv
Tasks:
  - Clean missing data
  - Encode Sex, Embarked columns
  - Visualize age distribution (Matplotlib/Seaborn)
  - Output cleaned dataset as new CSV
Author: Khushi Soni
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['font.size'] = 12

file_path = "Titanic-Dataset.csv"
try:
    df = pd.read_csv(file_path)
    print(f"\nSuccessfully loaded '{file_path}'")
    print(f"   Shape: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print(f"\nERROR: File '{file_path}' not found. Please check the path.")
    exit()

print("\nTASK 1: CLEANING MISSING DATA")
print("\nMissing values BEFORE cleaning:")
print(df.isnull().sum())

if df['Age'].isnull().sum() > 0:
    age_median = df['Age'].median()
    df['Age'].fillna(age_median, inplace=True)
    print(f"\nImputed 'Age' with MEDIAN = {age_median:.2f}")

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

print("\nMissing values AFTER cleaning:")
print(df.isnull().sum())
total_missing = df.isnull().sum().sum()
if total_missing == 0:
    print("All missing values have been handled successfully!")
else:
    print(f"Still {total_missing} missing values remain.")

print("\nTASK 2: ENCODING CATEGORICAL VARIABLES")
print("\n--- Encoding 'Sex' with LabelEncoder ---")
le = LabelEncoder()
df['Sex_encoded'] = le.fit_transform(df['Sex'])
sex_mapping = dict(zip(le.classes_, le.transform(le.classes_)))
print(f"   Mapping: {sex_mapping} (male→1, female→0)")

print("\n--- Encoding 'Embarked' with OneHotEncoder ---")
ohe = OneHotEncoder(sparse_output=False, drop='first')
embarked_encoded = ohe.fit_transform(df[['Embarked']])
feature_names = ohe.get_feature_names_out(['Embarked'])
embarked_df = pd.DataFrame(embarked_encoded, columns=feature_names, index=df.index)
print(f"   Created {len(feature_names)} new columns: {list(feature_names)}")

df_final = df.drop(columns=['Sex', 'Embarked'])
df_final = pd.concat([df_final, embarked_df], axis=1)

print("\nEncoding complete. First 5 rows of encoded dataset:")
print(df_final[['Sex_encoded'] + list(feature_names)].head())

print("\nTASK 3: VISUALIZING AGE DISTRIBUTION")
fig, axes = plt.subplots(1, 2, figsize=(14, 6))

sns.histplot(df['Age'], bins=20, kde=True, ax=axes[0], color='skyblue', edgecolor='black')
axes[0].set_title('Age Distribution (Histogram + KDE)', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Age', fontsize=12)
axes[0].set_ylabel('Frequency', fontsize=12)
axes[0].axvline(df['Age'].mean(), color='red', linestyle='--', label=f'Mean: {df["Age"].mean():.1f}')
axes[0].axvline(df['Age'].median(), color='green', linestyle='--', label=f'Median: {df["Age"].median():.1f}')
axes[0].legend()

sns.boxplot(y=df['Age'], ax=axes[1], color='lightgreen')
axes[1].set_title('Age Distribution (Boxplot)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Age', fontsize=12)
axes[1].grid(axis='y', linestyle='--', alpha=0.7)

fig.suptitle('Titanic Passenger Age Distribution', fontsize=16, fontweight='bold', y=1.02)

plt.tight_layout()
plt.savefig('age_distribution_plot.png', dpi=300, bbox_inches='tight')
print("Age distribution plot saved as 'age_distribution_plot.png'")

plt.show()

print("\nAge Statistics:")
print(f"   Mean age: {df['Age'].mean():.2f}")
print(f"   Median age: {df['Age'].median():.2f}")
print(f"   Min age: {df['Age'].min():.2f}")
print(f"   Max age: {df['Age'].max():.2f}")
print(f"   Standard deviation: {df['Age'].std():.2f}")

print("\nTASK 4: SAVING CLEANED DATASET")

output_file = "Titanic_cleaned_final.csv"
df_final.to_csv(output_file, index=False)
print(f"Cleaned dataset saved as '{output_file}'")
print(f" Final shape: {df_final.shape[0]} rows, {df_final.shape[1]} columns")
print(f" Final columns: {list(df_final.columns)}")
