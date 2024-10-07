import pandas as pd
from sklearn.metrics import (classification_report, confusion_matrix, roc_auc_score, matthews_corrcoef, precision_recall_curve, auc, balanced_accuracy_score)

df = pd.read_csv('predictions.csv')

# Extract true labels and predicted labels
y_true = df['transaction'].apply(lambda x: eval(x)['is_fraud'])
y_pred = df['prediction']

# Classification report (Precision, Recall, F1-Score)
print("Classification Report:")
print(classification_report(y_true, y_pred))

# Confusion matrix
conf_matrix = confusion_matrix(y_true, y_pred)
print(f"Confusion Matrix:\n{conf_matrix}")

# Matthews Correlation Coefficient (MCC)
mcc = matthews_corrcoef(y_true, y_pred)
print(f"Matthews Correlation Coefficient: {mcc}")

# ROC AUC score
roc_auc = roc_auc_score(y_true, y_pred)
print(f"ROC AUC Score: {roc_auc}")

# AUC-PR Score
precision, recall, _ = precision_recall_curve(y_true, y_pred)
pr_auc = auc(recall, precision)
print(f"Precision-Recall AUC Score: {pr_auc}")

# Balanced Accuracy Score
balanced_acc = balanced_accuracy_score(y_true, y_pred)
print(f"Balanced Accuracy Score: {balanced_acc}")
