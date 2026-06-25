# 📡 Telco Customer Churn Prediction

An end-to-end **Machine Learning project** for predicting customer churn using the **Telco Customer Churn Dataset**.

This project performs **Exploratory Data Analysis (EDA), Data Cleaning, Feature Engineering, Random Forest Classification, Hyperparameter Tuning, Cross-Validation, ROC–AUC Evaluation, and Customer Segmentation using K-Means Clustering**.

The objective is to identify customers likely to churn and create actionable customer segments for retention strategies.

---

## 📌 Project Overview

Customer churn prediction helps businesses identify customers who are likely to leave their service.

This project:

* Explores customer behavior patterns
* Cleans and preprocesses raw telecom data
* Builds predictive machine learning models
* Optimizes model performance
* Evaluates using classification metrics
* Segments customers into business-friendly groups

---

# 📁 Dataset

Dataset File:

```text
Telco_customer_churn.xlsx
```

### Target Variable

```text
Churn Value
```

* `1 → Customer Churned`
* `0 → Customer Retained`

### Important Features

* Tenure Months
* Monthly Charges
* Total Charges
* Contract Type
* Internet Service
* Payment Method
* Tech Support
* Customer Demographics

---

# 🛠️ Tech Stack

## Languages

* Python

## Libraries

```bash
pandas
numpy
matplotlib
seaborn
scikit-learn
openpyxl
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

---

# 📊 Step 1 — Exploratory Data Analysis (EDA)

EDA was performed to understand customer behavior and discover churn patterns.

### Analysis Performed

### 1. Churn Distribution

* Checked class imbalance
* Compared churned vs retained customers

### 2. Tenure Analysis

* Histogram + KDE
* Boxplot by churn

### 3. Monthly Charges

* Distribution analysis
* Quartile comparison

### 4. Contract Type

* Countplot with churn categories

### 5. Internet Service

* DSL vs Fiber Optic vs No Service

### 6. Payment Method

* Churn behavior across payment methods

### 7. Tech Support

* Relationship with churn probability

### 8. Correlation Heatmap

Compared:

```text
Tenure Months
Monthly Charges
Churn Value
Churn Score
CLTV
```

### 9. Cross Tabulation

Contract Type vs Churn Rate

---

# 🧹 Step 2 — Data Cleaning

### Total Charges Conversion

```python
df['Total Charges'] = pd.to_numeric(
    df['Total Charges'],
    errors='coerce'
)
```

### Missing Value Handling

* Missing values corresponded to customers with zero tenure.
* Replaced with:

```python
df['Total Charges'].fillna(0)
```

### Removed Columns

```python
drop_columns = [
'CustomerID',
'Count',
'Country',
'State',
'Zip Code',
'Lat Long',
'Latitude',
'City',
'Longitude',
'Churn Label',
'Churn Score',
'CLTV',
'Churn Reason'
]
```

---

# 🔢 Step 3 — Feature Encoding

One-Hot Encoding applied:

```python
df = pd.get_dummies(
    df,
    drop_first=True
)
```

Result:

* Converted categorical features into numeric values
* Expanded feature dimensions

---

# 🎯 Step 4 — Feature Selection

## Feature Matrix

```python
X = df.drop('Churn Value', axis=1)
Y = df['Churn Value']
```

### Low Importance Features Removed

```python
dropping = [
'Phone Service_Yes',
'Multiple Lines_No phone service',
'Streaming TV_Yes',
'Streaming Movies_Yes',
'Device Protection_No internet service'
]
```

Created:

```python
X_selected
```

---

# ✂️ Step 5 — Train Test Split

```python
from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(
X,
Y,
test_size=0.20,
random_state=42
)
```

Split Ratio:

```text
80% Training
20% Testing
```

---

# 🌲 Step 6 — Random Forest Modeling

## Baseline Model

```python
rf_model = RandomForestClassifier(
n_estimators=100,
random_state=42
)
```

Metrics:

* Accuracy
* Confusion Matrix
* Classification Report

---

## Approach I — Class Imbalance Handling

```python
rf_balanced = RandomForestClassifier(
n_estimators=100,
class_weight='balanced',
random_state=42
)
```

Purpose:

Improve churn detection recall.

---

## Approach II — Hyperparameter Tuning

```python
rf_tuned = RandomForestClassifier(
n_estimators=300,
max_depth=10,
class_weight='balanced',
random_state=42
)
```

Manual Grid Search:

```python
n_estimators_list = [
100,
200,
300,
400,
500
]

max_depth_list = [
5,
10,
15,
20,
25
]
```

Optimization Priority:

```text
Recall → Accuracy
```

---

## Approach III — Feature Importance

```python
feature_importance = pd.DataFrame({
'features': X.columns,
'importance': rf_tuned.feature_importances_
})
```

Actions:

* Ranked features
* Removed low-impact variables
* Retrained model

---

# 🔁 Step 7 — Cross Validation

```python
from sklearn.model_selection import cross_val_score

cv_accuracy = cross_val_score(
final_rf,
X,
Y,
cv=5,
scoring='accuracy'
)

cv_recall = cross_val_score(
final_rf,
X,
Y,
cv=5,
scoring='recall'
)
```

Performed:

* 5 Fold Validation
* Accuracy Mean
* Recall Mean

---

# 📈 Step 8 — ROC–AUC Evaluation

```python
from sklearn.metrics import (
roc_auc_score,
roc_curve
)
```

Probability Prediction:

```python
y_prob = rf_tuned.predict_proba(X_test)
```

Generated:

* ROC Curve
* AUC Score

Evaluation Goal:

Higher AUC = Better Separation

---

# 🗂️ Step 9 — Customer Segmentation (K-Means)

Segmentation Inputs:

```python
segmentation_data = pd.DataFrame({
'Tenure Months': X['Tenure Months'],
'Monthly Charges': X['Monthly Charges'],
'Total Charges': X['Total Charges'],
'Churn Probability': churn_probability
})
```

Scaling:

```python
StandardScaler()
```

### Elbow Method

```python
K = 3
```

---

## Customer Segments

| Cluster | Segment                |
| ------- | ---------------------- |
| 0       | Budget Loyal Customer  |
| 1       | High Risk Customer     |
| 2       | Loyal Premium Customer |

---

# 📉 Step 10 — Visualizations

Generated Visualizations:

* Tenure Distribution
* Tenure vs Churn
* Monthly Charges Distribution
* Monthly Charges vs Churn
* Contract Type Analysis
* Internet Service Analysis
* Payment Method Analysis
* Tech Support Analysis
* Correlation Heatmap
* Elbow Curve
* Cluster Scatterplots

---

# 🚀 How To Run

### Clone Repository

```bash
git clone <your-repository-url>
```

### Move Into Folder

```bash
cd Telco-Customer-Churn-Prediction
```

### Install Dependencies

```bash
pip install pandas numpy matplotlib seaborn scikit-learn openpyxl
```

### Start Notebook

```bash
jupyter notebook
```

Open:

```text
CBSOTPROJ1.ipynb
```

Run all cells sequentially.

---

# 📌 Key Findings

✅ Customers with shorter tenure are more likely to churn.

✅ Month-to-month contracts show higher churn.

✅ Higher monthly charges increase churn probability.

✅ Class balancing improved churn recall.

✅ Random Forest (300 Trees, Depth = 10) performed best.

✅ K-Means segmentation identified actionable customer groups.

---

# 📂 Project Structure

```text
Telco-Customer-Churn-Prediction
│
├── Telco_customer_churn.xlsx
├── CBSOTPROJ1.ipynb
├── README.md
├── requirements.txt
└── images/
```

---

# ⭐ Future Improvements

* Deploy using Streamlit
* Build Dashboard
* Add XGBoost & LightGBM
* Add SHAP Explainability
* Real-time churn prediction

---

# 👨‍💻 Author

**Vaishnavi Kushwah**

Machine Learning • AI • Data Analytics

---

If you found this project useful:

⭐ Star the repository
🍴 Fork the repository
📢 Share with others

---
