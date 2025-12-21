# Quick Start Guide

## Step-by-Step Instructions

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Generate the Dataset
```bash
python generate_dirty_data.py
```
This creates `customer_churn_dirty.csv` with 12,000+ rows and intentional data quality issues.

### 3. Run the Project

**Option A: Run as Python Script**
```bash
python ml_final_project.py
```

**Option B: Run as Jupyter Notebook**
```bash
jupyter notebook ml_final_project.ipynb
```

## What the Project Includes

✅ **Part 1**: Problem domain explanation (Customer Churn & CLV Prediction)

✅ **Part 2**: Data collection (12,000+ rows, 10+ columns with dirty data)

✅ **Part 3**: Complete data cleaning (all 10 steps):
- Missing values handling
- Duplicate removal
- Outlier detection and removal
- Blank space handling
- Inconsistent data entry standardization
- Simple Imputer usage

✅ **Part 4**: Exploratory Data Analysis with correlation analysis

✅ **Part 5**: 8 Types of Visualizations:
1. Line plots
2. Area plots
3. Histogram
4. Bar charts
5. Pie charts
6. Box plots
7. Scatter plots
8. Bubble plots

✅ **Part 6**: Model Building:
- **Classification**: 4 models (Logistic Regression, Random Forest, Gradient Boosting, SVM)
  - Metrics: Accuracy, Precision, Recall, ROC-AUC
- **Regression**: 4 models (Linear Regression, Random Forest, Gradient Boosting, SVR)
  - Metrics: MAE, MSE, Median AE, R² Score
- Scaling effects tested
- Normalization effects tested
- Result visualizations

## Output Files

All visualizations are saved in the `visualizations/` folder:
- 8 basic visualization types
- Model results comparison
- Confusion matrix
- ROC curves

## Notes

- The dataset is synthetically generated with intentional issues for learning purposes
- All code is well-commented and organized
- Both script and notebook versions are available
- Ready for submission with all requirements met

