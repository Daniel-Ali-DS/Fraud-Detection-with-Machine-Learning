import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from xgboost import XGBClassifier
from sklearn.metrics import classification_report
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv('fraud_data_pandas.csv')

# Drop unnecessary columns 
X = df.drop(['is_fraud', 'trans_timestamp'], axis=1)
y = df['is_fraud']

# Define categorical and numeric features for encoding
categorical_features = ['age_group', 'merchant', 'city']
numeric_features = ['total_amount', 'distance_km']

# ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ])

# Preprocess the data BEFORE applying SMOTE
X_preprocessed = preprocessor.fit_transform(X)

# Apply SMOTE to handle class imbalance (now X is numeric)
smote = SMOTE(random_state=42)
X_smote, y_smote = smote.fit_resample(X_preprocessed, y)

# Split data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_smote, y_smote, test_size=0.3, random_state=42)

# Train the XGBoost classifier directly after preprocessing
xgb_model = XGBClassifier(n_estimators=300, learning_rate=0.1, max_depth=6, random_state=42)
xgb_model.fit(X_train, y_train)

# Get predicted probabilities from the model
y_pred_proba = xgb_model.predict_proba(X_test)

# Assign new labels based on thresholds
new_labels = np.where(y_pred_proba[:, 1] > 0.75, 2, np.where(y_pred_proba[:, 1] > 0.5, 1, 0))

# Evaluate the multiclass model
print("Classification Report for Multiclass:")
print(classification_report(y_test, new_labels))
