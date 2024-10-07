import numpy as np
import pandas as pd
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, matthews_corrcoef, roc_auc_score, precision_recall_curve, auc

# Load the data
data = pd.read_csv('fraud_data_pandas.csv')

# Preprocessing
label_encoder = LabelEncoder()
data['merchant'] = label_encoder.fit_transform(data['merchant'])
data['city'] = label_encoder.fit_transform(data['city'])
data['age_group'] = label_encoder.fit_transform(data['age_group'])

# Features and target
X = data.drop(columns=['is_fraud', 'trans_timestamp'])
y = data['is_fraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# Standardize the features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Train XGBoost model
xgb = XGBClassifier(n_estimators=750, learning_rate=0.3, eval_metric='auc', scale_pos_weight=(y_train == 0).sum() / (y_train == 1).sum())
xgb.fit(X_train, y_train)

# Train TensorFlow model
tf_model = tf.keras.Sequential([
    tf.keras.Input(shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(32, activation='relu'),
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Compile the TensorFlow model
tf_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc')])

# Train the TensorFlow model
tf_model.fit(X_train_scaled, y_train, epochs=10, batch_size=64, validation_data=(X_test_scaled, y_test))

# Get Predictions from both models
xgb_pred_proba = xgb.predict_proba(X_test)[:, 1]  # XGBoost predictions
tf_pred_proba = tf_model.predict(X_test_scaled).flatten()  # TensorFlow predictions

# Combine the predictions into a new dataset
stacked_predictions = np.column_stack((xgb_pred_proba, tf_pred_proba))

# Train a meta-model (e.g., Logistic Regression)
meta_model = LogisticRegression()
meta_model.fit(stacked_predictions, y_test)

# Make final predictions using the meta-model
final_predictions = meta_model.predict(stacked_predictions)

# Evaluate the final stacked model
print("Classification Report for Stacked Model:")
print(classification_report(y_test, final_predictions))

mcc = matthews_corrcoef(y_test, final_predictions)
print(f"Matthews Correlation Coefficient (Stacked): {mcc}")

auc_roc = roc_auc_score(y_test, meta_model.predict_proba(stacked_predictions)[:, 1])
print(f"AUC-ROC Score (Stacked): {auc_roc}")

precision, recall, _ = precision_recall_curve(y_test, meta_model.predict_proba(stacked_predictions)[:, 1])
auc_pr = auc(recall, precision)
print(f"AUC-PR Score (Stacked): {auc_pr}")
