
# ============================================
# STEP 4: ADDITIONAL DATASET INSPECTION
# ============================================

import pandas as pd
import os

# Dataset path
file_path = "dataset/additional/Dataset_5971.csv"

# Check file
if not os.path.exists(file_path):
    raise FileNotFoundError("Additional dataset not found!")

# Load CSV
df = pd.read_csv(file_path)

# 1. Dataset shape
print("\n========== DATASET SHAPE ==========")
print(df.shape)

# 2. Column names
print("\n========== COLUMN NAMES ==========")
print(df.columns.tolist())

# 3. First 10 records
print("\n========== FIRST 10 RECORDS ==========")
print(df.head(10).to_string())

# 4. Dataset information
print("\n========== DATASET INFO ==========")
df.info()

# 5. Missing values
print("\n========== MISSING VALUES ==========")
print(df.isnull().sum())

# 6. Duplicate records
print("\n========== DUPLICATE RECORDS ==========")
print(df.duplicated().sum())

# 7. Unique values of each column
print("\n========== UNIQUE VALUES ==========")

for column in df.columns:
    print(f"\n{column}:")
    print(df[column].value_counts().head(20))

print("\nDATASET INSPECTION COMPLETED!")
