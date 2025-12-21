# Machine Learning Final Project

## Customer Churn Prediction & Customer Lifetime Value Prediction

This project implements a complete machine learning pipeline covering data collection, cleaning, exploratory data analysis, visualization, and model building for both classification and regression tasks.

## Project Structure

```
machineleraningfinalproject/
├── README.md
├── requirements.txt
├── generate_dirty_data.py          # Script to generate dirty dataset
├── ml_final_project.py             # Main project script
├── ml_final_project.ipynb          # Jupyter notebook version
├── customer_churn_dirty.csv        # Generated dataset (after running generate_dirty_data.py)
└── visualizations/                 # Folder containing all visualization outputs
    ├── 1_line_plot.png
    ├── 2_area_plot.png
    ├── 3_histogram.png
    ├── 4_bar_chart.png
    ├── 5_pie_chart.png
    ├── 6_box_plot.png
    ├── 7_scatter_plot.png
    ├── 8_bubble_plot.png
    ├── model_results.png
    └── confusion_matrix.png
```

## Installation

### Option 1: Using Virtual Environment (Recommended)

1. Create a virtual environment:
```bash
python -m venv .venv
```

2. Activate the virtual environment:
   - **Windows (PowerShell):**
     ```bash
     .venv\Scripts\Activate.ps1
     ```
   - **Windows (Command Prompt):**
     ```bash
     .venv\Scripts\activate.bat
     ```
   - **Linux/Mac:**
     ```bash
     source .venv/bin/activate
     ```

3. Install required packages:
```bash
pip install -r requirements.txt
```

### Option 2: Global Installation

1. Install required packages directly:
```bash
pip install -r requirements.txt
```

## Usage

### Step 1: Generate the Dirty Dataset

First, run the data generation script to create a dataset with intentional data quality issues:

```bash
python generate_dirty_data.py
```

This will create `customer_churn_dirty.csv` with:
- 12,000+ rows
- 10+ columns (mix of numerical and categorical)
- Intentional issues: missing values, duplicates, outliers, inconsistent data entry, etc.

### Step 2: Run the Main Project

#### Option A: Run as Python Script
```bash
python ml_final_project.py
```

#### Option B: Run as Jupyter Notebook
```bash
jupyter notebook ml_final_project.ipynb
```

## Project Components

### Part 1: Problem Domain
- **Classification Task**: Predict customer churn (Yes/No)
- **Regression Task**: Predict customer lifetime value

### Part 2: Data Collection
- Dataset with 12,000+ rows and 10+ columns
- Mix of numerical and categorical features
- Intentional data quality issues for cleaning practice

### Part 3: Data Cleaning (10 Steps)
1. ✅ Deal with Missing Values
2. ✅ Figure out why data is missing
3. ✅ Eliminate extra variables
4. ✅ Eliminate duplicates
5. ✅ Detect and remove outliers (using IQR method)
6. ✅ Scaling and Normalization (tested during modeling)
7. ✅ Eliminate blank spaces (using Simple Imputer)
8. ✅ Arrange data logically
9. ✅ Group data in rows and columns
10. ✅ Deal with inconsistent data entry

### Part 4: Exploratory Data Analysis
- Correlation analysis
- Variable relationship exploration
- Statistical summaries

### Part 5: Visualization (8 Types)
1. ✅ Line plots
2. ✅ Area plots
3. ✅ Histogram
4. ✅ Bar charts
5. ✅ Pie charts
6. ✅ Box plots
7. ✅ Scatter plots
8. ✅ Bubble plots

### Part 6: Model Building

#### Classification Models (Churn Prediction)
1. Logistic Regression
2. Random Forest Classifier
3. Gradient Boosting Classifier
4. Support Vector Machine (SVM)

**Evaluation Metrics:**
- Accuracy
- Precision
- Recall
- ROC-AUC Score

#### Regression Models (Customer Lifetime Value)
1. Linear Regression
2. Random Forest Regressor
3. Gradient Boosting Regressor
4. Support Vector Regression (SVR)

**Evaluation Metrics:**
- Mean Absolute Error (MAE)
- Mean Squared Error (MSE)
- Median Absolute Error
- R² Score

#### Additional Analysis
- ✅ Effect of scaling on model performance
- ✅ Effect of normalization on model performance
- ✅ Visualization of model results

## Features

- **Age**: Customer age
- **Monthly Charges**: Monthly service charges
- **Total Charges**: Total charges accumulated
- **Tenure Months**: Number of months as customer
- **Monthly Usage GB**: Data usage per month
- **Customer Satisfaction**: Satisfaction score (1-10)
- **Number of Services**: Number of services subscribed
- **Gender**: Customer gender
- **Contract Type**: Type of contract
- **Payment Method**: Payment method used
- **Internet Service**: Type of internet service
- **Phone Service**: Whether phone service is active
- **Streaming TV**: Whether streaming TV is active
- **Streaming Movies**: Whether streaming movies is active

## Output

The project generates:
- Cleaned dataset
- 8 different types of visualizations
- Model performance metrics and comparisons
- ROC curves and confusion matrices
- Analysis of scaling and normalization effects

## Requirements

- Python 3.8+
- pandas
- numpy
- matplotlib
- seaborn
- scikit-learn
- jupyter (optional, for notebook)

## Notes

- The dataset is generated synthetically with intentional data quality issues
- All models are trained and evaluated with proper train/test splits
- Results are saved in the `visualizations/` folder
- The project follows all requirements from the AIE 121 ML Final Project specification




