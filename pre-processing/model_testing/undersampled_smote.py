import pandas as pd
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, roc_auc_score, precision_recall_curve, auc, matthews_corrcoef
from imblearn.over_sampling import SMOTE

# Load dataset
df = pd.read_csv('fraud_data_pandas.csv')

# Drop unnecessary columns for the model (adjust based on dataset)
X = df.drop(['is_fraud', 'trans_timestamp'], axis=1)
y = df['is_fraud']

# Define categorical and numeric features for encoding
categorical_features = ['age_group']
numeric_features = ['total_amount', 'distance_km']
categorical_ordinal = ['merchant', 'city']

# Create the ColumnTransformer for preprocessing
preprocessor = ColumnTransformer(
    transformers=[
        ('cat_ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_ordinal),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ])

# Initial undersampling of non-fraud transactions for 70/30 class balance
fraud = df[df['is_fraud'] == 1]
non_fraud = df[df['is_fraud'] == 0]

# Check the number of fraudulent transactions
fraud_count = len(fraud)

# Take 20,000 non-fraudulent and all available fraudulent transactions (if fewer than 3,000)
non_fraud_sample = non_fraud.sample(n=20000, random_state=42)
fraud_sample = fraud.sample(n=min(fraud_count, 3000), random_state=42)

# Merge back to create a balanced dataset
balanced_df = pd.concat([non_fraud_sample, fraud_sample])

# Separate into features (X) and target (y)
X_balanced = balanced_df.drop(['is_fraud', 'trans_timestamp'], axis=1)
y_balanced = balanced_df['is_fraud']

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.3, random_state=42)

# Create a pipeline with preprocessing and the XGBoost model
xgb_model = XGBClassifier(
    n_estimators=300,
    learning_rate=0.1,
    max_depth=6,
    min_child_weight=1,
    scale_pos_weight=1,  # Adjust for class imbalance
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42
)

# Create the pipeline with preprocessing and model
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('classifier', xgb_model)
])

# Fit the model on the training data
pipeline.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = pipeline.predict(X_test)
y_pred_proba = pipeline.predict_proba(X_test)[:, 1]
print("Initial Model Performance:")
print(classification_report(y_test, y_pred))

# Preprocess the full data for SMOTE
X_preprocessed = preprocessor.fit_transform(X)

# SMOTE
smote = SMOTE(random_state=42)
X_resampled, y_resampled = smote.fit_resample(X_preprocessed, y)

# Train a new model
pipeline_smote = Pipeline(steps=[
    ('classifier', XGBClassifier(
        n_estimators=600,
        learning_rate=0.2,
        min_child_weight=1,
        scale_pos_weight=1,
        subsample=0.8,
        eval_metric='auc',
        colsample_bytree=0.8,
        random_state=42
    ))
])

# Since SMOTE is applied after preprocessing, we don't need the preprocessor in the pipeline now
pipeline_smote.fit(X_resampled, y_resampled)
y_pred_smote = pipeline_smote.predict(preprocessor.transform(X_test))
print("SMOTE Model Performance:")
print(classification_report(y_test, y_pred_smote))

# Metrics and Cross-Validation
print("\n---- Additional Metrics ----")

# Cross-validation to evaluate model
cv_scores = cross_val_score(pipeline, X_balanced, y_balanced, cv=5, scoring='roc_auc')
print("Cross-validation AUC-ROC scores for each fold:", cv_scores)
print("Mean AUC-ROC from cross-validation:", cv_scores.mean())

# AUC-ROC score
auc_roc = roc_auc_score(y_test, y_pred_proba)
print(f"AUC-ROC Score: {auc_roc}")

# Precision-Recall curve and AUC-PR
precision, recall, _ = precision_recall_curve(y_test, y_pred_proba)
auc_pr = auc(recall, precision)
print(f"AUC-PR Score: {auc_pr}")

# Matthews Correlation Coefficient
mcc = matthews_corrcoef(y_test, y_pred)
print(f"Matthews Correlation Coefficient: {mcc}")

# Feature Importance
feature_importance = pipeline.named_steps['classifier'].feature_importances_
print("Feature Importance: ", feature_importance)

# Printing feature names
feature_names = preprocessor.get_feature_names_out()
print("Feature Names: ", feature_names)
