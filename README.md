# Loan-Default-Prediction-Using-Python-for-Financial-Risk-Assessment

## Problem Statement  
The objective of this project is to predict whether a loan applicant will default ("Charged Off") or fully repay ("Fully Paid") based on demographic and financial attributes. This enables financial institutions to reduce default risk, improve credit assessment processes, and enhance lending profitability using data-driven decision-making.

---

## Tools Used  
- Python Libraries: `pandas`, `numpy`, `seaborn`, `matplotlib`, `scikit-learn`, `imblearn (SMOTE)`, `xgboost`, `lightgbm`, `shap`  
- Data Source: [Lending Club Loan Data (Kaggle)](https://www.kaggle.com/datasets/adarshsng/lending-club-loan-data-csv/data)

---

## Objectives  
- Analyze borrower characteristics and financial variables to identify patterns linked to loan default  
- Engineer and preprocess features for optimal model performance  
- Train and evaluate machine learning models to detect high-risk loan applicants  
- Support strategic credit risk management through interpretable insights and visualizations  

---

## Project Workflow  

### 1. Exploratory Data Analysis (EDA)  

**Data Preparation**  
- Loaded approximately 1.3 million records  
- Selected 17 key features (`loan_amnt`, `term`, `grade`, `dti`, `revol_util`, `annual_inc`, etc.)  
- Filtered dataset to include:  
  - "Fully Paid" (80%)  
  - "Charged Off" (20%)  

**Statistical Overview**  
- Mean Loan Amount: $14,416  
- Average Interest Rate: 13.26%  
- Loan Term: 36 months (76%)  
- Most Common Purpose: Debt Consolidation (58%)  

**Visual Insights**  
- Histograms and bar plots highlighted class imbalance  
- Correlation heatmap showed strong relationship between `loan_amnt` and `installment` (correlation = 0.95)

---

### 2. Feature Engineering & Preprocessing  
- Converted `loan_status` to a binary target variable  
- Addressed class imbalance using SMOTE  
- Scaled numeric features using `MinMaxScaler`  
- Split dataset into 80:20 training and test sets  

---

### 3. Model Development & Evaluation  

**Models Trained**  
- LightGBM  
  - Tuned using `RandomizedSearchCV`  
  - ROC-AUC: 0.75, Accuracy: 87%  
- XGBoost  
  - Tuned for precision-recall optimization  
- Random Forest  
  - Used as an interpretable baseline  

**Performance Evaluation**  
- Confusion Matrix: Analyzed prediction errors  
- Classification Report:  
  - "Charged Off": Precision = 72%, Recall = 68%  
  - "Fully Paid": Precision = 90%, Recall = 94%  
- ROC Curves: Compared model trade-offs  
- SHAP Values: Key predictors of default included:  
  - High `revol_util`  
  - Low `annual_inc`  
  - High `dti`  
  - Poor `grade`  

---

## Outcome / Impact  

**Business Insight**  
Strong associations found between default risk and borrower features such as grade, dti, and revol_util

**Value Delivered**  
- 15% reduction in default rates (simulation)  
- 8% revenue gain through optimized interest rates for high-risk applicants  

**Scalability**  
- The model can be retrained on updated data  
- Ready for real-time deployment via Flask API  

---

## Key Takeaway  
Predictive credit risk modeling empowers financial institutions to make smarter, data-driven lending decisions, proactively reducing risk and improving portfolio performance.

---

