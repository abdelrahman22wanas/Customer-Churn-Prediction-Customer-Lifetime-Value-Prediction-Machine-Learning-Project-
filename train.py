"""
AIE 121 - Machine Learning Final Project
Training script: trains models, saves artifacts for API deployment.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler, LabelEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, \
    GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.svm import SVC, SVR
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, roc_auc_score,
    roc_curve, mean_absolute_error, mean_squared_error,
    median_absolute_error, r2_score, confusion_matrix
)
import joblib
import os
import warnings
warnings.filterwarnings('ignore')

sns.set_style("whitegrid")
plt.rcParams['figure.figsize'] = (12, 6)

MODEL_DIR = "models"
VISUALIZATION_DIR = "visualizations"
os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(VISUALIZATION_DIR, exist_ok=True)

print("=" * 80)
print("AIE 121 - Training Pipeline (Deployment Ready)")
print("=" * 80)

# ---------------------------------------------------------------------------
# DATA LOADING
# ---------------------------------------------------------------------------
try:
    df = pd.read_csv('customer_churn_dirty.csv')
    print(f"Dataset loaded: {df.shape[0]} rows, {df.shape[1]} columns")
except FileNotFoundError:
    print("Run 'generate_dirty_data.py' first.")
    exit(1)

# ---------------------------------------------------------------------------
# DATA CLEANING
# ---------------------------------------------------------------------------
df = df.drop_duplicates()

numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
for col in numerical_cols:
    if col in df.columns:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        df = df[(df[col] >= lower) | (df[col].isna())]
        df = df[(df[col] <= upper) | (df[col].isna())]

categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].replace([' ', ''], np.nan)
        df[col] = df[col].apply(
            lambda x: np.nan if isinstance(x, str) and x.strip() == '' else x
        )

num_imputer = SimpleImputer(strategy='median')
for col in numerical_cols:
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col] = num_imputer.fit_transform(df[[col]]).ravel()

cat_imputer = SimpleImputer(strategy='most_frequent')
for col in categorical_cols:
    if col in df.columns and df[col].isnull().sum() > 0:
        df[col] = cat_imputer.fit_transform(df[[col]]).ravel()

df = df.sort_values(by=['tenure_months', 'age']).reset_index(drop=True)

for col in categorical_cols:
    if col in df.columns:
        df[col] = df[col].astype(str).str.lower().str.strip()
        if 'churn' in col.lower():
            df[col] = df[col].replace({'yes': 'yes', 'y': 'yes', 'no': 'no', 'n': 'no'})
        elif 'gender' in col.lower():
            df[col] = df[col].replace({'male': 'male', 'm': 'male', 'female': 'female', 'f': 'female'})
        elif 'yes' in df[col].values or 'no' in df[col].values:
            df[col] = df[col].replace({'yes': 'yes', 'y': 'yes', 'no': 'no', 'n': 'no'})

df['monthly_charges'] = df['monthly_charges'].abs()
df['age'] = df['age'].abs()
df['total_charges'] = df['total_charges'].abs()

print(f"Cleaned dataset: {df.shape[0]} rows, {df.shape[1]} columns")

# ---------------------------------------------------------------------------
# VISUALIZATIONS (kept from original)
# ---------------------------------------------------------------------------
# 1. Line Plot
monthly_charges_by_tenure = df.groupby('tenure_months')['monthly_charges'].mean().sort_index()
plt.figure(figsize=(12, 6))
plt.plot(monthly_charges_by_tenure.index, monthly_charges_by_tenure.values, marker='o', linewidth=2)
plt.xlabel('Tenure (Months)')
plt.ylabel('Average Monthly Charges')
plt.title('Line Plot: Average Monthly Charges by Tenure')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{VISUALIZATION_DIR}/1_line_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 2. Area Plot
churn_by_contract = df.groupby('contract_type')['churn'].value_counts().unstack(fill_value=0)
plt.figure(figsize=(12, 6))
churn_by_contract.plot(kind='area', stacked=True, alpha=0.7)
plt.xlabel('Contract Type')
plt.ylabel('Number of Customers')
plt.title('Area Plot: Churn Distribution by Contract Type')
plt.legend(title='Churn')
plt.tight_layout()
plt.savefig(f'{VISUALIZATION_DIR}/2_area_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 3. Histogram
plt.figure(figsize=(12, 6))
plt.hist(df['age'], bins=30, edgecolor='black', alpha=0.7, color='skyblue')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.title('Histogram: Age Distribution')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{VISUALIZATION_DIR}/3_histogram.png', dpi=300, bbox_inches='tight')
plt.close()

# 4. Bar Chart
plt.figure(figsize=(12, 6))
churn_counts = df['churn'].value_counts()
plt.bar(churn_counts.index, churn_counts.values, color=['#ff6b6b', '#4ecdc4'], alpha=0.7)
plt.xlabel('Churn')
plt.ylabel('Count')
plt.title('Bar Chart: Churn Distribution')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig(f'{VISUALIZATION_DIR}/4_bar_chart.png', dpi=300, bbox_inches='tight')
plt.close()

# 5. Pie Chart
plt.figure(figsize=(10, 8))
contract_counts = df['contract_type'].value_counts()
plt.pie(contract_counts.values, labels=contract_counts.index, autopct='%1.1f%%', startangle=90)
plt.title('Pie Chart: Contract Type Distribution')
plt.tight_layout()
plt.savefig(f'{VISUALIZATION_DIR}/5_pie_chart.png', dpi=300, bbox_inches='tight')
plt.close()

# 6. Box Plot
plt.figure(figsize=(12, 6))
df.boxplot(column=['monthly_charges', 'total_charges', 'tenure_months'], grid=True, figsize=(12, 6))
plt.ylabel('Value')
plt.title('Box Plot: Distribution of Numerical Features')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(f'{VISUALIZATION_DIR}/6_box_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 7. Scatter Plot
plt.figure(figsize=(12, 6))
churn_colors = {'yes': 'red', 'no': 'blue'}
colors = df['churn'].map(churn_colors)
plt.scatter(df['monthly_charges'], df['total_charges'], c=colors, alpha=0.5, s=50)
plt.xlabel('Monthly Charges')
plt.ylabel('Total Charges')
plt.title('Scatter Plot: Monthly Charges vs Total Charges (colored by Churn)')
plt.legend(handles=[
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='red', markersize=10, label='Churn: Yes'),
    plt.Line2D([0], [0], marker='o', color='w', markerfacecolor='blue', markersize=10, label='Churn: No')
])
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{VISUALIZATION_DIR}/7_scatter_plot.png', dpi=300, bbox_inches='tight')
plt.close()

# 8. Bubble Plot
plt.figure(figsize=(12, 8))
churn_yes = df[df['churn'] == 'yes']
churn_no = df[df['churn'] == 'no']
plt.scatter(churn_yes['age'], churn_yes['monthly_charges'],
            s=churn_yes['tenure_months'] * 5, alpha=0.5, c='red', label='Churn: Yes')
plt.scatter(churn_no['age'], churn_no['monthly_charges'],
            s=churn_no['tenure_months'] * 5, alpha=0.5, c='blue', label='Churn: No')
plt.xlabel('Age')
plt.ylabel('Monthly Charges')
plt.title('Bubble Plot: Age vs Monthly Charges (bubble size = tenure)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{VISUALIZATION_DIR}/8_bubble_plot.png', dpi=300, bbox_inches='tight')
plt.close()

print("Visualizations saved to 'visualizations/' folder.")

# ---------------------------------------------------------------------------
# FEATURE ENGINEERING
# ---------------------------------------------------------------------------
feature_cols = [
    'age', 'monthly_charges', 'total_charges', 'tenure_months',
    'monthly_usage_gb', 'customer_satisfaction', 'number_of_services',
    'gender', 'contract_type', 'payment_method', 'internet_service',
    'phone_service', 'streaming_tv', 'streaming_movies'
]

X = df[feature_cols].copy()
y_classification = df['churn'].copy()
y_regression = df['customer_lifetime_value'].copy()

categorical_features = X.select_dtypes(include=['object']).columns.tolist()
numerical_features = X.select_dtypes(include=[np.number]).columns.tolist()

label_encoders = {}
X_encoded = X.copy()
for col in categorical_features:
    le = LabelEncoder()
    X_encoded[col] = le.fit_transform(X[col].astype(str))
    label_encoders[col] = le

le_target = LabelEncoder()
y_classification_encoded = le_target.fit_transform(y_classification)

X_final = X_encoded.values

# ---------------------------------------------------------------------------
# TRAIN / TEST SPLIT
# ---------------------------------------------------------------------------
X_train, X_test, y_train_clf, y_test_clf = train_test_split(
    X_final, y_classification_encoded, test_size=0.2, random_state=42,
    stratify=y_classification_encoded
)
X_train_reg, X_test_reg, y_train_reg, y_test_reg = train_test_split(
    X_final, y_regression, test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------------
# TRAIN CLASSIFICATION MODELS
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TRAINING CLASSIFICATION MODELS (Churn Prediction)")
print("=" * 80)

classification_models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42),
    'SVM': SVC(probability=True, random_state=42)
}

classification_results = {}
for name, model in classification_models.items():
    print(f"\n--- {name} ---")
    model.fit(X_train, y_train_clf)
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None

    accuracy = accuracy_score(y_test_clf, y_pred)
    precision = precision_score(y_test_clf, y_pred, average='weighted')
    recall = recall_score(y_test_clf, y_pred, average='weighted')
    roc_auc = roc_auc_score(y_test_clf, y_pred_proba) if y_pred_proba is not None else None

    classification_results[name] = {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'roc_auc': roc_auc,
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba
    }
    print(f"  Accuracy:  {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall:    {recall:.4f}")
    if roc_auc:
        print(f"  ROC-AUC:   {roc_auc:.4f}")

# ---------------------------------------------------------------------------
# TRAIN REGRESSION MODELS
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("TRAINING REGRESSION MODELS (Customer Lifetime Value)")
print("=" * 80)

regression_models = {
    'Linear Regression': LinearRegression(),
    'Random Forest': RandomForestRegressor(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingRegressor(n_estimators=100, random_state=42),
    'SVR': SVR()
}

regression_results = {}
for name, model in regression_models.items():
    print(f"\n--- {name} ---")
    model.fit(X_train_reg, y_train_reg)
    y_pred = model.predict(X_test_reg)

    mae = mean_absolute_error(y_test_reg, y_pred)
    mse = mean_squared_error(y_test_reg, y_pred)
    median_ae = median_absolute_error(y_test_reg, y_pred)
    r2 = r2_score(y_test_reg, y_pred)

    regression_results[name] = {
        'model': model,
        'mae': mae,
        'mse': mse,
        'median_ae': median_ae,
        'r2': r2,
        'y_pred': y_pred
    }
    print(f"  MAE:  {mae:.4f}")
    print(f"  MSE:  {mse:.4f}")
    print(f"  MedAE:{median_ae:.4f}")
    print(f"  R²:   {r2:.4f}")

# ---------------------------------------------------------------------------
# PICK BEST MODELS
# ---------------------------------------------------------------------------
best_clf_name = max(classification_results, key=lambda n: classification_results[n]['accuracy'])
best_reg_name = max(regression_results, key=lambda n: regression_results[n]['r2'])

best_clf = classification_results[best_clf_name]
best_reg = regression_results[best_reg_name]

print(f"\nBest Classification Model: {best_clf_name} "
      f"(Accuracy: {best_clf['accuracy']:.4f})")
print(f"Best Regression Model:    {best_reg_name} "
      f"(R²: {best_reg['r2']:.4f})")

# ---------------------------------------------------------------------------
# SCALING / NORMALIZATION COMPARISON (model results visualization)
# ---------------------------------------------------------------------------
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)
X_train_reg_scaled = scaler.fit_transform(X_train_reg)
X_test_reg_scaled = scaler.transform(X_test_reg)

normalizer = MinMaxScaler()
X_train_norm = normalizer.fit_transform(X_train)
X_test_norm = normalizer.transform(X_test)
X_train_reg_norm = normalizer.fit_transform(X_train_reg)
X_test_reg_norm = normalizer.transform(X_test_reg)

for label, X_tr, X_te in [("Scaled", X_train_scaled, X_test_scaled),
                           ("Normalized", X_train_norm, X_test_norm)]:
    print(f"\n--- Classification with {label} ---")
    for name in classification_models:
        m = type(classification_models[name])(**classification_models[name].get_params())
        m.fit(X_tr, y_train_clf)
        acc = accuracy_score(y_test_clf, m.predict(X_te))
        print(f"  {name}: {acc:.4f}")

    print(f"\n--- Regression with {label} ---")
    for name in regression_models:
        m = type(regression_models[name])(**regression_models[name].get_params())
        m.fit(X_tr_reg := (X_train_reg_scaled if "Scaled" in label else X_train_reg_norm),
              y_train_reg)
        pred = m.predict(X_te_reg := (X_test_reg_scaled if "Scaled" in label else X_test_reg_norm))
        r2 = r2_score(y_test_reg, pred)
        print(f"  {name}: R² = {r2:.4f}")

# ---------------------------------------------------------------------------
# MODEL RESULTS VISUALIZATION
# ---------------------------------------------------------------------------
plt.figure(figsize=(14, 10))

plt.subplot(2, 2, 1)
models = list(classification_results.keys())
accuracies = [classification_results[m]['accuracy'] for m in models]
precisions = [classification_results[m]['precision'] for m in models]
recalls = [classification_results[m]['recall'] for m in models]
x = np.arange(len(models))
width = 0.25
plt.bar(x - width, accuracies, width, label='Accuracy', alpha=0.8)
plt.bar(x, precisions, width, label='Precision', alpha=0.8)
plt.bar(x + width, recalls, width, label='Recall', alpha=0.8)
plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Classification Metrics Comparison')
plt.xticks(x, models, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(2, 2, 2)
for name, results in classification_results.items():
    if results['y_pred_proba'] is not None:
        fpr, tpr, _ = roc_curve(y_test_clf, results['y_pred_proba'])
        plt.plot(fpr, tpr, label=f"{name} (AUC={results['roc_auc']:.3f})")
plt.plot([0, 1], [0, 1], 'k--', label='Random')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curves')
plt.legend()
plt.grid(True, alpha=0.3)

plt.subplot(2, 2, 3)
reg_models = list(regression_results.keys())
r2_scores = [regression_results[m]['r2'] for m in reg_models]
mae_scores = [regression_results[m]['mae'] for m in reg_models]
mae_normalized = [m / max(mae_scores) for m in mae_scores]
x = np.arange(len(reg_models))
width = 0.35
plt.bar(x - width / 2, r2_scores, width, label='R² Score', alpha=0.8)
plt.bar(x + width / 2, mae_normalized, width, label='MAE (normalized)', alpha=0.8)
plt.xlabel('Models')
plt.ylabel('Score')
plt.title('Regression Metrics Comparison')
plt.xticks(x, reg_models, rotation=45, ha='right')
plt.legend()
plt.grid(True, alpha=0.3, axis='y')

plt.subplot(2, 2, 4)
best_pred = best_reg['y_pred']
plt.scatter(y_test_reg, best_pred, alpha=0.5)
plt.plot([y_test_reg.min(), y_test_reg.max()],
         [y_test_reg.min(), y_test_reg.max()], 'r--', lw=2)
plt.xlabel('Actual CLV')
plt.ylabel('Predicted CLV')
plt.title(f'Actual vs Predicted ({best_reg_name})')
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{VISUALIZATION_DIR}/model_results.png', dpi=300, bbox_inches='tight')
plt.close()

# Confusion Matrix
plt.figure(figsize=(8, 6))
cm = confusion_matrix(y_test_clf, best_clf['y_pred'])
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['No Churn', 'Churn'],
            yticklabels=['No Churn', 'Churn'])
plt.ylabel('Actual')
plt.xlabel('Predicted')
plt.title(f'Confusion Matrix - {best_clf_name}')
plt.tight_layout()
plt.savefig(f'{VISUALIZATION_DIR}/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

print("\nModel result visualizations saved!")

# ---------------------------------------------------------------------------
# SAVE ARTIFACTS
# ---------------------------------------------------------------------------
print("\n" + "=" * 80)
print("SAVING MODEL ARTIFACTS")
print("=" * 80)

joblib.dump(best_clf['model'], f'{MODEL_DIR}/classifier.pkl')
joblib.dump(best_reg['model'], f'{MODEL_DIR}/regressor.pkl')
joblib.dump(label_encoders, f'{MODEL_DIR}/label_encoders.pkl')
joblib.dump(le_target, f'{MODEL_DIR}/target_encoder.pkl')
joblib.dump(scaler, f'{MODEL_DIR}/scaler.pkl')
joblib.dump(normalizer, f'{MODEL_DIR}/normalizer.pkl')
joblib.dump(numerical_features, f'{MODEL_DIR}/numerical_features.pkl')
joblib.dump(categorical_features, f'{MODEL_DIR}/categorical_features.pkl')
joblib.dump(feature_cols, f'{MODEL_DIR}/feature_columns.pkl')

print(f"  classifier.pkl           ({best_clf_name})")
print(f"  regressor.pkl            ({best_reg_name})")
print(f"  label_encoders.pkl       ({len(label_encoders)} encoders)")
print(f"  target_encoder.pkl       (churn labels)")
print(f"  scaler.pkl / normalizer.pkl")
print(f"  numerical_features.pkl   ({len(numerical_features)} features)")
print(f"  categorical_features.pkl ({len(categorical_features)} features)")
print(f"  feature_columns.pkl      ({len(feature_cols)} columns)")
print(f"\nAll artifacts saved to '{MODEL_DIR}/'")

print("\n" + "=" * 80)
print("TRAINING COMPLETE")
print("=" * 80)
