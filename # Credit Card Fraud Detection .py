# Credit Card Fraud Detection: Sample Code Structure
# -----------------------------------------------
# Technologies Used: Python, pandas, numpy, scikit-learn, imbalanced-learn, XGBoost, matplotlib, seaborn

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, IsolationForest
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load Data
data = pd.read_csv('creditcard.csv')

# 2. Data Exploration & Preprocessing
print(data.info())
print(data['Class'].value_counts())  # Class: 0 = legitimate, 1 = fraud

# Feature Scaling
scaler = StandardScaler()
data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))

# 3. Split Data
X = data.drop(['Class', 'Time'], axis=1)
y = data['Class']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# 4. Handle Imbalanced Data using SMOTE
sm = SMOTE(random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# 5. Model Training (Random Forest)
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_res, y_res)

# 6. Prediction & Evaluation
y_pred = model.predict(X_test)
print(classification_report(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))

# ROC Curve
y_pred_prob = model.predict_proba(X_test)[:, 1]
fpr, tpr, thresholds = roc_curve(y_test, y_pred_prob)
plt.figure(figsize=(8, 6))
plt.plot(fpr, tpr, label='Random Forest')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve')
plt.legend()
plt.show()

# 7. Anomaly Detection Example (Isolation Forest)
iso_forest = IsolationForest(contamination=0.001, random_state=42)
y_iso = iso_forest.fit_predict(X_test)
# -1: anomaly, 1: normal
print(pd.Series(y_iso).value_counts())