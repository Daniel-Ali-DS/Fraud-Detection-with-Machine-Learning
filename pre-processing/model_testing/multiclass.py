import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler, OrdinalEncoder
from xgboost import XGBClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, roc_auc_score, matthews_corrcoef, precision_recall_curve, auc

# Load the dataset
df_pandas = pd.read_csv('fraud_data_pandas.csv')

# Drop unnecessary columns
X = df_pandas.drop(columns=['is_fraud', 'trans_timestamp'])
y = df_pandas['is_fraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42, stratify=y)

# Adjust scale_pos_weight (reduced from 258.14)
non_fraud_count = len(y_train[y_train == 0])
fraud_count = len(y_train[y_train == 1])
scale_pos_weight = non_fraud_count / fraud_count
print(f"Scale Pos Weight: {scale_pos_weight}")

# Define categorical and numeric features for encoding
categorical_features = ['age_group']
numeric_features = ['total_amount', 'distance_km']
categorical_ordinal = ['merchant', 'city']

# Preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('cat_ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_ordinal),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ])

# Fine-tune XGBoost (lower learning rate, higher estimators, regularization)
xg_classifier = XGBClassifier(
    n_estimators=1000,  # Increased number of estimators
    learning_rate=0.05,  # Lower learning rate for finer learning
    scale_pos_weight=scale_pos_weight,
    reg_alpha=0.5,  # L1 regularization
    reg_lambda=0.5,  # L2 regularization
    random_state=42,
    eval_metric='logloss'
)

# Build the pipeline
xgb_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xg_classifier)
])

# Fit the pipeline
xgb_pipeline.fit(X_train, y_train)

# Cross-validation to evaluate model
cv_scores = cross_val_score(xgb_pipeline, X_train, y_train, cv=5, scoring='roc_auc')
print("Cross-validation AUC-ROC scores for each fold:", cv_scores)
print("Mean AUC-ROC from cross-validation:", cv_scores.mean())

# Experiment with different thresholds
def evaluate_with_threshold(threshold):

    # Predict probabilities
    y_pred_proba = xgb_pipeline.predict_proba(X_test)[:, 1]

    # Apply threshold
    y_pred = (y_pred_proba >= threshold).astype(int)

    # Classification report
    print(f"\nClassification Report (Threshold: {threshold}):")
    print(classification_report(y_test, y_pred))

    # Matthews Correlation Coefficient
    mcc = matthews_corrcoef(y_test, y_pred)
    print(f"Matthews Correlation Coefficient (Threshold {threshold}): {mcc}")

    # AUC-ROC and AUC-PR
    auc_roc = roc_auc_score(y_test, y_pred_proba)
    print(f"AUC-ROC Score (Threshold {threshold}): {auc_roc}")

    precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
    auc_pr = auc(recall, precision)
    print(f"AUC-PR Score (Threshold {threshold}): {auc_pr}")


# Evaluate with different thresholds
thresholds = [0.5, 0.6, 0.7]
for threshold in thresholds:
    evaluate_with_threshold(threshold)

# Feature Importance
print("Feature Importance: ", xg_classifier.feature_importances_)
