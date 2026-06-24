# ❤️ Heart Disease Prediction using Machine Learning

## 📌 Project Overview

This project builds a Machine Learning model to predict whether a person is at risk of heart disease based on their medical information.

The project uses the UCI Heart Disease dataset and performs the complete Machine Learning pipeline:

* Data Cleaning
* Exploratory Data Analysis (EDA)
* Data Preprocessing
* Feature Engineering
* Logistic Regression Model Training
* Model Evaluation
* Feature Importance Analysis

This project demonstrates a real-world binary classification workflow often used in healthcare analytics.

---

## 🎯 Objective

Predict whether a patient has heart disease.

Original dataset target:

* `0` = No heart disease
* `1,2,3,4` = Different severity levels of heart disease

For this project, the problem was converted into binary classification:

* `0` → No heart disease
* `1` → Heart disease

---

## 🛠️ Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Seaborn
* Scikit-learn

---

## 📂 Dataset Information

Dataset: Heart Disease UCI Dataset

Features:

| Feature  | Description                 |
| -------- | --------------------------- |
| age      | Age of patient              |
| sex      | Gender                      |
| dataset  | Dataset source              |
| cp       | Chest pain type             |
| trestbps | Resting blood pressure      |
| chol     | Cholesterol                 |
| fbs      | Fasting blood sugar         |
| restecg  | Resting ECG                 |
| thalch   | Maximum heart rate achieved |
| exang    | Exercise induced angina     |
| oldpeak  | ST depression               |
| slope    | Slope of peak exercise      |
| ca       | Number of major vessels     |
| thal     | Thalassemia                 |
| num      | Original disease severity   |

---

## 🚀 Project Workflow

### 1. Data Loading

* Loaded CSV dataset
* Inspected shape, columns and datatypes

### 2. Data Cleaning

Performed:

* Missing value handling
* Removed unnecessary columns
* Converted target variable to binary classification

### 3. Exploratory Data Analysis (EDA)

Visualizations created:

* Heart disease distribution
* Age distribution
* Age vs heart disease
* Gender analysis
* Correlation heatmap

### 4. Feature Engineering

Performed One Hot Encoding on categorical features.

### 5. Data Splitting

Dataset split:

* 80% Training
* 20% Testing

### 6. Feature Scaling

Used StandardScaler.

### 7. Model Training

Model used:

Logistic Regression

### 8. Model Evaluation

Metrics used:

* Accuracy
* Classification Report
* Confusion Matrix
* ROC Curve
* ROC-AUC Score

### 9. Feature Importance Analysis

Logistic Regression coefficients were analyzed to determine influential features.

---

## 📊 Evaluation Metrics

### Accuracy

Measures overall prediction correctness.

### Confusion Matrix

Shows:

* True Positives
* True Negatives
* False Positives
* False Negatives

### ROC Curve

Shows model discrimination capability.

### ROC-AUC

Measures the ability of the model to distinguish between classes.

Interpretation:

* 0.5 → Poor
* 0.7 → Acceptable
* 0.8 → Good
* 0.9+ → Excellent

---

## 📁 Project Structure

```text
heart_disease_project/

│

├── heart.csv

├── heart_disease.ipynb

├── README.md
```

---

## ▶️ Installation

Clone repository:

```bash
git clone https://github.com/yourusername/heart-disease-prediction.git
```

Install dependencies:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
```

Run notebook:

```bash
jupyter notebook
```

---

## 💡 Future Improvements

Possible enhancements:

* Compare multiple models

  * Decision Tree
  * Random Forest
  * XGBoost

* Hyperparameter tuning

* Save model using Joblib

* Build a Streamlit web application

* Deploy the model

---

## 📚 Learning Outcomes

This project helped understand:

* Binary Classification
* Medical Data Analysis
* Data Cleaning
* EDA
* Feature Engineering
* Logistic Regression
* ROC-AUC Evaluation
* Feature Importance Analysis

---

## 👨‍💻 Author

Sardar Ahsan

Machine Learning | Python | Data Science | AI Enthusiast
