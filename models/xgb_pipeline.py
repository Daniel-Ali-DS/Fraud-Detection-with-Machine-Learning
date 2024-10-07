import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelEncoder, FunctionTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, matthews_corrcoef, precision_recall_curve, auc, confusion_matrix
import pickle
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('fraud_data_pandas.csv')

# Encode categorical features
label_encoder_merchant = LabelEncoder()
label_encoder_city = LabelEncoder()
label_encoder_age_group = LabelEncoder()

data['merchant'] = label_encoder_merchant.fit_transform(data['merchant'])
data['city'] = label_encoder_city.fit_transform(data['city'])
data['age_group'] = label_encoder_age_group.fit_transform(data['age_group'])

# Separate features and target
X = data.drop(columns=['is_fraud', 'trans_timestamp'])
y = data['is_fraud']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# Calculate scale_pos_weight for class imbalance
scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()
print(f"Scale Pos Weight: {scale_pos_weight}")

# Initialize XGBoost classifier
xgb_model = XGBClassifier(scale_pos_weight=scale_pos_weight, n_estimators=750, learning_rate=0.3, eval_metric='auc')

# Fit the model on the training data
xgb_model.fit(X_train, y_train)

# Save the pipeline, including the LabelEncoders and model
pipeline_dict = {
    'model': xgb_model,
    'label_encoder_merchant': label_encoder_merchant,
    'label_encoder_city': label_encoder_city,
    'label_encoder_age_group': label_encoder_age_group
}

# Save pipeline as .pkl file
with open('fraud_detection_pipeline.pkl', 'wb') as f:
    pickle.dump(pipeline_dict, f)

print("Pipeline saved as fraud_detection_pipeline.pkl")

# Predict probabilities and apply custom threshold
y_proba = xgb_model.predict_proba(X_test)[:, 1]
confidence_threshold = 0.7
y_pred = (y_proba >= confidence_threshold).astype(int)

# Feature importance
print("Feature Importance: ", xgb_model.feature_importances_)

# Model performance evaluation
print(f"Using Confidence Threshold: {confidence_threshold}")
print(classification_report(y_test, y_pred))

# Calculate Matthews Correlation Coefficient
mcc = matthews_corrcoef(y_test, y_pred)
print(f"Matthews Correlation Coefficient: {mcc}")

# Calculate AUC-ROC score
auc_roc = roc_auc_score(y_test, y_proba)
print(f"AUC-ROC Score: {auc_roc}")

# Calculate Precision-Recall and AUC-PR
precision, recall, _ = precision_recall_curve(y_test, y_proba)
auc_pr = auc(recall, precision)
print(f"AUC-PR Score: {auc_pr}")

# Confusion Matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print(f"Confusion Matrix: \n{conf_matrix}")

# Visualizing Confusion Matrix using Seaborn heatmap
plt.figure(figsize=(6,4))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', cbar=False, xticklabels=['Not Fraud', 'Fraud'], yticklabels=['Not Fraud', 'Fraud'])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()
