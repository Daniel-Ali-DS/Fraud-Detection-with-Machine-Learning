import pandas as pd
import tensorflow as tf
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import matthews_corrcoef, roc_auc_score, precision_recall_curve, auc, classification_report, accuracy_score
from tensorflow.keras.callbacks import ReduceLROnPlateau
from scikeras.wrappers import KerasClassifier

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

# Define the learning rate reduction callback
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=3, min_lr=1e-6)

# Define a function to create the Keras model
def create_model(optimizer='adam', learning_rate=0.001):
    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(X_train_scaled.shape[1],)),
        tf.keras.layers.Dense(256, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.001)),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dropout(0.2),
        tf.keras.layers.Dense(32, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    optimizer_fn = tf.keras.optimizers.Adam(learning_rate=learning_rate) if optimizer == 'adam' else tf.keras.optimizers.SGD(learning_rate=learning_rate)
    model.compile(optimizer=optimizer_fn, loss='binary_crossentropy', metrics=[tf.keras.metrics.AUC(name='auc'), 'accuracy'])
    return model

# Wrap the model using KerasClassifier for scikit-learn compatibility
keras_clf = KerasClassifier(model=create_model, optimizer='adam', learning_rate=0.001, epochs=10, batch_size=64, class_weight={0: 1., 1: 12.}, callbacks=[reduce_lr])

# Define the hyperparameter grid for GridSearch
param_grid = {
    'optimizer': ['adam', 'sgd'],
    'learning_rate': [0.001, 0.0001],
    'batch_size': [32, 64],
    'epochs': [10, 20]
}

# Set up GridSearchCV
grid = GridSearchCV(estimator=keras_clf, param_grid=param_grid, scoring='roc_auc', cv=3)

# Fit GridSearchCV
grid_result = grid.fit(X_train_scaled, y_train)

# Output best parameters and results
print(f"Best parameters: {grid_result.best_params_}")
print(f"Best ROC-AUC: {grid_result.best_score_}")

# Evaluate the best model
best_model = grid_result.best_estimator_.model_
y_pred_proba = best_model.predict(X_test_scaled).flatten()
y_pred = (y_pred_proba >= 0.7).astype(int)

# Print the classification report
print("Classification Report for Best Model:")
print(classification_report(y_test, y_pred, zero_division=1))

# Calculate Matthews Correlation Coefficient
mcc = matthews_corrcoef(y_test, y_pred)
print(f"Matthews Correlation Coefficient: {mcc}")

# AUC-ROC and AUC-PR scores
auc_roc = roc_auc_score(y_test, y_pred_proba)
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
auc_pr = auc(recall, precision)
print(f"AUC-ROC Score: {auc_roc}")
print(f"AUC-PR Score: {auc_pr}")
