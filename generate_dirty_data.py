"""
Script to generate a dirty dataset for the ML Final Project
This creates a dataset with intentional issues: missing values, duplicates, outliers, etc.

Dependencies: pandas, numpy (install via: pip install -r requirements.txt)
"""
import pandas as pd
import numpy as np
import random

np.random.seed(42)
random.seed(42)

n_samples = 12000

# ---------------------------------------------------------------------------
# 1. Generate CLEAN base features
# ---------------------------------------------------------------------------
age = np.random.randint(18, 80, n_samples)
monthly_charges = np.random.normal(65, 20, n_samples).clip(10, 200)
tenure_months = np.random.randint(1, 72, n_samples)
total_charges = monthly_charges * tenure_months * np.random.uniform(0.8, 1.2, n_samples)
monthly_usage_gb = np.random.normal(50, 25, n_samples).clip(0, 500)
customer_satisfaction = np.random.randint(1, 11, n_samples)
number_of_services = np.random.randint(1, 5, n_samples)

contract_clean = np.random.choice(
    ['month-to-month', 'one year', 'two year'], n_samples, p=[0.5, 0.25, 0.25]
)
gender_clean = np.random.choice(['male', 'female'], n_samples)
payment_clean = np.random.choice(
    ['electronic check', 'mailed check', 'bank transfer', 'credit card'],
    n_samples, p=[0.4, 0.2, 0.2, 0.2]
)
internet_clean = np.random.choice(
    ['fiber optic', 'dsl', 'no'], n_samples, p=[0.4, 0.35, 0.25]
)
phone_clean = np.random.choice(['yes', 'no'], n_samples, p=[0.7, 0.3])
streaming_tv_clean = np.random.choice(['yes', 'no'], n_samples)
streaming_movies_clean = np.random.choice(['yes', 'no'], n_samples)

# ---------------------------------------------------------------------------
# 2. Generate TARGETS from features (realistic business logic)
# ---------------------------------------------------------------------------

# Churn probability via logistic function
# Higher churn: month-to-month, low satisfaction, high charges, short tenure, few services, young age
# Lower churn:  two-year contract, high satisfaction, long tenure, phone service
churn_logit = (
    - 0.2
    + 1.8 * (contract_clean == 'month-to-month').astype(float)
    - 1.5 * (contract_clean == 'two year').astype(float)
    - 0.35 * customer_satisfaction
    + 0.015 * monthly_charges
    - 0.03 * tenure_months
    + 0.3 * (number_of_services <= 1).astype(float)
    - 0.3 * (phone_clean == 'yes').astype(float)
    + 0.15 * (internet_clean == 'fiber optic').astype(float)
    + 0.2 * (age < 30).astype(float)
)

churn_prob = 1 / (1 + np.exp(-churn_logit))
churn_binary = np.random.binomial(1, churn_prob)
churn_labels = np.where(churn_binary == 1, 'yes', 'no')

# CLV derived from features and churn status
# Higher CLV: high charges, long tenure, many services, high satisfaction, long contract
# Lower CLV: churned customers
base_clv = (
    + 15.0 * monthly_charges
    + 25.0 * tenure_months
    + 100.0 * number_of_services
    + 80.0 * customer_satisfaction
    + 200.0 * (contract_clean == 'two year').astype(float)
    + 50.0 * (phone_clean == 'yes').astype(float)
    - 2000.0 * churn_binary
    + np.random.normal(0, 800, n_samples)
).clip(100, 30000)

customer_lifetime_value = np.round(base_clv, 2)

# ---------------------------------------------------------------------------
# 3. Assemble DataFrame with CLEAN data
# ---------------------------------------------------------------------------
df = pd.DataFrame({
    'age': age,
    'monthly_charges': monthly_charges,
    'total_charges': total_charges,
    'tenure_months': tenure_months,
    'monthly_usage_gb': monthly_usage_gb,
    'customer_satisfaction': customer_satisfaction,
    'number_of_services': number_of_services,
    'gender': gender_clean,
    'contract_type': contract_clean,
    'payment_method': payment_clean,
    'internet_service': internet_clean,
    'phone_service': phone_clean,
    'streaming_tv': streaming_tv_clean,
    'streaming_movies': streaming_movies_clean,
    'churn': churn_labels,
    'customer_lifetime_value': customer_lifetime_value,
})

print(f"Churn rate: {churn_binary.mean():.3f}")
print(f"CLV mean:   {customer_lifetime_value.mean():.0f}, std: {customer_lifetime_value.std():.0f}")

# ---------------------------------------------------------------------------
# 4. Apply DATA QUALITY ISSUES (dirty up the features only, targets stay clean)
# ---------------------------------------------------------------------------

# 4a. Inconsistent casing and abbreviations
dirty_cats = {
    'gender': {
        'male': ['Male', 'male', 'M'],
        'female': ['Female', 'female', 'F'],
    },
    'contract_type': {
        'month-to-month': ['Month-to-month', 'monthly'],
        'one year': ['One year', 'yearly'],
        'two year': ['Two year'],
    },
    'payment_method': {
        'electronic check': ['Electronic check', 'electronic'],
        'mailed check': ['Mailed check'],
        'bank transfer': ['Bank transfer'],
        'credit card': ['Credit card', 'credit'],
    },
    'internet_service': {
        'fiber optic': ['Fiber optic', 'fiber'],
        'dsl': ['DSL', 'dsl'],
        'no': ['No', 'None'],
    },
    'phone_service': {
        'yes': ['Yes', 'yes', 'Y'],
        'no': ['No', 'no', 'N'],
    },
    'streaming_tv': {'yes': ['Yes', 'yes'], 'no': ['No', 'no']},
    'streaming_movies': {'yes': ['Yes', 'yes'], 'no': ['No', 'no']},
}

for col, mapping in dirty_cats.items():
    for clean_val, dirty_vals in mapping.items():
        mask = df[col] == clean_val
        n_dirty = mask.sum()
        if n_dirty > 0:
            df.loc[mask, col] = np.random.choice(dirty_vals, size=n_dirty)

# 4b. Missing values in numerical columns
missing_idx = np.random.choice(df.index, size=int(0.15 * len(df)), replace=False)
for col in ['age', 'monthly_charges', 'total_charges', 'tenure_months']:
    col_missing = np.random.choice(missing_idx, size=int(0.1 * len(missing_idx)), replace=False)
    df.loc[col_missing, col] = np.nan

# 4c. Missing values in categorical columns
for col in ['gender', 'contract_type', 'payment_method']:
    col_missing = np.random.choice(missing_idx, size=int(0.05 * len(missing_idx)), replace=False)
    df.loc[col_missing, col] = np.nan

# 4d. Duplicate rows
dup_rows = df.sample(n=500, random_state=42)
df = pd.concat([df, dup_rows], ignore_index=True)

# 4e. Outliers
outlier_idx = np.random.choice(df.index, size=200, replace=False)
df.loc[outlier_idx, 'monthly_charges'] = np.random.uniform(200, 500, 200)
df.loc[outlier_idx, 'total_charges'] = np.random.uniform(10000, 50000, 200)
df.loc[outlier_idx, 'age'] = np.random.choice([100, 150, 200], 200)

# 4f. Blank spaces
blank_idx = np.random.choice(df.index, size=100, replace=False)
df.loc[blank_idx, 'gender'] = ' '
df.loc[blank_idx, 'contract_type'] = ' '

# 4g. Negative values
neg_idx = np.random.choice(df.index, size=50, replace=False)
df.loc[neg_idx, 'monthly_charges'] = -abs(df.loc[neg_idx, 'monthly_charges'])
df.loc[neg_idx, 'age'] = -abs(df.loc[neg_idx, 'age'])

# Shuffle
df = df.sample(frac=1, random_state=42).reset_index(drop=True)

df.to_csv('customer_churn_dirty.csv', index=False)
print(f"Generated dirty dataset with {len(df)} rows and {len(df.columns)} columns")
print(f"Missing values: {df.isnull().sum().sum()}")
print(f"Duplicates: {df.duplicated().sum()}")
