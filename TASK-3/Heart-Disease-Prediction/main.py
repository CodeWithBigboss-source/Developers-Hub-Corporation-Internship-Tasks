# ==================================
# HEART DISEASE PREDICTION PROJECT
# ==================================

# -------- 1. Import libraries --------

import pandas as pd
import numpy as np

import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler

from sklearn.linear_model import LogisticRegression

from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    roc_auc_score
)

# -------- 2. Load dataset --------

df = pd.read_csv("heart_disease_uci.csv")

print("Dataset Loaded Successfully")

print(df.head())

print("\nShape:")

print(df.shape)

print("\nColumns:")

print(df.columns)


# -------- 3. Create Binary Target --------

# 0 = No disease
# 1,2,3,4 = Disease

df['target'] = (df['num'] > 0).astype(int)

# Remove original num column

df.drop(columns=['num'], inplace=True)


# -------- 4. Remove ID --------

df.drop(columns=['id'], inplace=True)


# -------- 5. Handle Missing Values --------

numeric_cols = df.select_dtypes(
    include=['int64','float64']
).columns

categorical_cols = df.select_dtypes(
    include=['object','bool']
).columns


# Fill numeric missing values

for col in numeric_cols:

    df[col] = df[col].fillna(
        df[col].median()
    )


# Fill categorical missing values

for col in categorical_cols:

    df[col] = df[col].fillna(
        df[col].mode()[0]
    )


print("\nMissing Values After Cleaning")

print(df.isnull().sum())


# -------- 6. EDA --------

# Target Distribution

plt.figure(figsize=(6,4))

sns.countplot(
    x='target',
    data=df
)

plt.title("Heart Disease Distribution")

plt.show()


# Age Distribution

plt.figure(figsize=(6,4))

sns.histplot(
    df['age'],
    bins=20
)

plt.title("Age Distribution")

plt.show()


# Age vs Disease

plt.figure(figsize=(6,4))

sns.boxplot(
    x='target',
    y='age',
    data=df
)

plt.title("Age vs Disease")

plt.show()


# Gender Analysis

plt.figure(figsize=(6,4))

sns.countplot(
    x='sex',
    hue='target',
    data=df
)

plt.title("Gender Analysis")

plt.show()


# -------- 7. One Hot Encoding --------

df = pd.get_dummies(

    df,

    drop_first=True

)


# -------- 8. Correlation Heatmap --------

plt.figure(figsize=(18,12))

sns.heatmap(

    df.corr(),

    cmap='coolwarm'

)

plt.title("Correlation Heatmap")

plt.show()


# -------- 9. Features and Target --------

X = df.drop(

    'target',

    axis=1

)

y = df['target']


# -------- 10. Train Test Split --------

X_train, X_test, y_train, y_test = train_test_split(

    X,

    y,

    test_size=0.2,

    random_state=42,

    stratify=y

)


# -------- 11. Feature Scaling --------

scaler = StandardScaler()

X_train = scaler.fit_transform(

    X_train

)

X_test = scaler.transform(

    X_test

)


# -------- 12. Train Logistic Regression --------

model = LogisticRegression(

    max_iter=5000

)

model.fit(

    X_train,

    y_train

)


# -------- 13. Predictions --------

y_pred = model.predict(

    X_test

)

y_prob = model.predict_proba(

    X_test

)[:,1]


# -------- 14. Accuracy --------

accuracy = accuracy_score(

    y_test,

    y_pred

)

print("\nAccuracy")

print(round(accuracy,3))


# -------- 15. Classification Report --------

print("\nClassification Report")

print(

classification_report(

y_test,

y_pred

)

)


# -------- 16. Confusion Matrix --------

cm = confusion_matrix(

    y_test,

    y_pred

)

print("\nConfusion Matrix")

print(cm)


plt.figure(figsize=(6,4))

sns.heatmap(

    cm,

    annot=True,

    fmt='d'

)

plt.xlabel("Predicted")

plt.ylabel("Actual")

plt.title("Confusion Matrix")

plt.show()


# -------- 17. ROC Curve --------

auc = roc_auc_score(

    y_test,

    y_prob

)

print("\nROC AUC")

print(round(auc,3))


fpr,tpr,thresholds = roc_curve(

    y_test,

    y_prob

)


plt.figure(figsize=(6,4))

plt.plot(

    fpr,

    tpr,

    label=f"AUC={auc:.2f}"

)

plt.plot(

    [0,1],

    [0,1]

)

plt.xlabel("False Positive Rate")

plt.ylabel("True Positive Rate")

plt.title("ROC Curve")

plt.legend()

plt.show()


# -------- 18. Feature Importance --------

feature_names = X.columns

importance = pd.DataFrame({

'Feature':feature_names,

'Importance':model.coef_[0]

})

importance = importance.sort_values(

    by='Importance',

    ascending=False

)

print("\nTop Important Features")

print(

importance.head(10)

)


plt.figure(figsize=(12,8))

sns.barplot(

    data=importance.head(15),

    x='Importance',

    y='Feature'

)

plt.title("Top Features")

plt.show()


# -------- 19. Final Summary --------

print("\n========== PROJECT SUMMARY ==========")

print("Rows:",df.shape[0])

print("Columns:",df.shape[1])

print("Accuracy:",round(accuracy,3))

print("ROC AUC:",round(auc,3))