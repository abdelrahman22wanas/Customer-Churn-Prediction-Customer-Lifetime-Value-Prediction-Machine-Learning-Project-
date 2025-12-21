"""
Script to generate a dirty dataset for the ML Final Project
This creates a dataset with intentional issues: missing values, duplicates, outliers, etc.

Dependencies: pandas, numpy (install via: pip install -r requirements.txt)
"""
import pandas as pd  # type: ignore
import numpy as np  # type: ignore
import random

# Set random seed for reproducibility
np.random.seed(42)
random.seed(42)

# Generate base data
n_samples = 12000  # More than 10000 as required

# Create dirty dataset with intentional issues
data = {
    # Numerical columns
    'age': np.random.randint(18, 80, n_samples),
    'monthly_charges': np.random.normal(65, 20, n_samples),
    'total_charges': np.random.normal(2000, 1000, n_samples),
    'tenure_months': np.random.randint(1, 72, n_samples),
    'monthly_usage_gb': np.random.normal(50, 25, n_samples),
    'customer_satisfaction': np.random.randint(1, 11, n_samples),
    'number_of_services': np.random.randint(1, 5, n_samples),
    
    # Categorical columns
    'gender': np.random.choice(['Male', 'Female', 'male', 'female', 'M', 'F'], n_samples),
    'contract_type': np.random.choice(['Month-to-month', 'One year', 'Two year', 'monthly', 'yearly'], n_samples),
    'payment_method': np.random.choice(['Electronic check', 'Mailed check', 'Bank transfer', 'Credit card', 'electronic', 'credit'], n_samples),
    'internet_service': np.random.choice(['DSL', 'Fiber optic', 'No', 'None', 'dsl', 'fiber'], n_samples),
    'phone_service': np.random.choice(['Yes', 'No', 'yes', 'no', 'Y', 'N'], n_samples),
    'streaming_tv': np.random.choice(['Yes', 'No', 'yes', 'no'], n_samples),
    'streaming_movies': np.random.choice(['Yes', 'No', 'yes', 'no'], n_samples),
    
    # Target variables
    'churn': np.random.choice(['Yes', 'No', 'yes', 'no'], n_samples),
    'customer_lifetime_value': np.random.normal(5000, 2000, n_samples)
}

df = pd.DataFrame(data)

# Introduce intentional data quality issues

# 1. Missing values (randomly introduce NaN)
missing_indices = np.random.choice(df.index, size=int(0.15 * len(df)), replace=False)
for col in ['age', 'monthly_charges', 'total_charges', 'tenure_months']:
    col_missing = np.random.choice(missing_indices, size=int(0.1 * len(missing_indices)), replace=False)
    df.loc[col_missing, col] = np.nan

# Missing values in categorical columns
for col in ['gender', 'contract_type', 'payment_method']:
    col_missing = np.random.choice(missing_indices, size=int(0.05 * len(missing_indices)), replace=False)
    df.loc[col_missing, col] = np.nan

# 2. Duplicates (introduce some duplicate rows)
duplicate_rows = df.sample(n=500, random_state=42)
df = pd.concat([df, duplicate_rows], ignore_index=True)

# 3. Outliers (introduce extreme values)
outlier_indices = np.random.choice(df.index, size=200, replace=False)
df.loc[outlier_indices, 'monthly_charges'] = np.random.uniform(200, 500, 200)
df.loc[outlier_indices, 'total_charges'] = np.random.uniform(10000, 50000, 200)
df.loc[outlier_indices, 'age'] = np.random.choice([100, 150, 200], 200)

# 4. Inconsistent data entry (already done in categorical columns with different cases)

# 5. Blank spaces
blank_indices = np.random.choice(df.index, size=100, replace=False)
df.loc[blank_indices, 'gender'] = ' '
df.loc[blank_indices, 'contract_type'] = ' '

# 6. Negative values where they shouldn't exist
negative_indices = np.random.choice(df.index, size=50, replace=False)
df.loc[negative_indices, 'monthly_charges'] = -abs(df.loc[negative_indices, 'monthly_charges'])
df.loc[negative_indices, 'age'] = -abs(df.loc[negative_indices, 'age'])

# Shuffle the dataframe
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

# Save to CSV
df.to_csv('customer_churn_dirty.csv', index=False)
print(f"Generated dirty dataset with {len(df)} rows and {len(df.columns)} columns")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")

