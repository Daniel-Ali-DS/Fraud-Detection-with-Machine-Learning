import pandas as pd
import numpy as np
import tensorflow as tf 
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import matthews_corrcoef, roc_auc_score, precision_recall_curve, auc, classification_report, accuracy_score
from tensorflow.keras.callbacks import ReduceLROnPlateau
from tensorflow.keras import Input

# Load the dataset
data = pd.read_csv('fraud_data_pandas.csv')

# Preprocessing
label_encoder = LabelEncoder()
data['merchant'] = label_encoder.fit_transform(data['merchant'])
data['city'] = label_encoder.fit_transform(data['city'])
data['age_group'] = label_encoder.fit_transform(data['age_group'])

# Drop 'trans_timestamp' and prepare features (X) and target (y)
X = data.drop(columns=['is_fraud', 'trans_timestamp'])
y = data['is_fraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.15, random_state=42, stratify=y)

# Convert y_train and y_test to numpy arrays
y_train = y_train.values
y_test = y_test.values

# Normalize numeric data
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Define the TensorFlow/Keras model based on best hyperparameters
model = tf.keras.Sequential([
    Input(shape=(X_train_scaled.shape[1],)),
    tf.keras.layers.Dense(32, activation='relu'),  # Best units from Keras Tuner
    tf.keras.layers.Dense(32, activation='relu'),  # Best number of layers from Keras Tuner
    tf.keras.layers.Dense(1, activation='sigmoid')
])

# Reduce learning rate callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)

# Custom optimizer with the best learning rate from Keras Tuner
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0011)

# Compile the TensorFlow model
model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc'), 'accuracy'])

# Class weights to address class imbalance (fraud class is 12x more important)
class_weight = {0: 1., 1: 12.}

# Train the TensorFlow model with class_weight
history = model.fit(X_train_scaled, y_train, class_weight=class_weight, epochs=10, batch_size=64, validation_data=(X_test_scaled, y_test), callbacks=[reduce_lr])

# Predict probabilities for TensorFlow model
confidence_threshold = 0.7  # Example threshold
tf_proba = model.predict(X_test_scaled).flatten()  # Probabilities
tf_pred = (tf_proba >= confidence_threshold).astype(int)  # Convert probabilities to class predictions

# Evaluate the TensorFlow model
print("Classification Report for TensorFlow model:")
print(classification_report(y_test, tf_pred, zero_division=1))

# Calculate other metrics
mcc = matthews_corrcoef(y_test, tf_pred)
print(f"Matthews Correlation Coefficient (TensorFlow): {mcc}")

# AUC-ROC and AUC-PR scores
auc_roc = roc_auc_score(y_test, tf_proba)
precision, recall, _ = precision_recall_curve(y_test, tf_proba)
auc_pr = auc(recall, precision)
print(f"AUC-ROC Score (TensorFlow): {auc_roc}")
print(f"AUC-PR Score (TensorFlow): {auc_pr}")