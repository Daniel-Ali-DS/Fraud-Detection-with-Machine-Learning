import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import OrdinalEncoder, OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import warnings

warnings.filterwarnings('ignore')

# Load the dataset
df = pd.read_csv("fraud_data_pandas.csv")

# Drop unnecessary column
df = df.drop('trans_timestamp', axis=1)

# Define categorical and numeric features for encoding
categorical_features = ['age_group']
numeric_features = ['total_amount', 'distance_km']
categorical_ordinal = ['merchant', 'city']

# Create the ColumnTransformer for preprocessing (Label and OneHotEncoding)
preprocessor = ColumnTransformer(
    transformers=[
        ('cat_ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1), categorical_ordinal),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features),
        ('num', StandardScaler(), numeric_features)
    ])

# Apply preprocessing to the features (except target column 'is_fraud')
df_encoded = df.copy()
X = df_encoded.drop(columns=['is_fraud'])
y = df_encoded['is_fraud']
X_encoded = preprocessor.fit_transform(X)

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, train_size=0.8, test_size=0.2, random_state=1)

# Build the XGBoost model
xg = xgb.XGBClassifier()

# Train the model
xg.fit(X_train, y_train)

# Confusion Matrix - Model performance
def RunModel(model, X_train, y_train, X_test, y_test):
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    matrix = confusion_matrix(y_test, pred)
    return matrix, pred


# Run the model and evaluate it
cmat, pred = RunModel(xg, X_train, y_train, X_test, y_test)

# Display the confusion matrix
cm = confusion_matrix(y_test, pred)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()

# Model accuracy and classification report
print(f"Accuracy Score: {accuracy_score(y_test, pred)}")
print(classification_report(y_test, pred))

# Downsampling technique to balance the dataset (undersampling the majority class)
fraud_records = len(df_encoded[df_encoded.is_fraud == 1])
fraud_indices = df_encoded[df_encoded.is_fraud == 1].index
normal_indices = df_encoded[df_encoded.is_fraud == 0].index

# Randomly select normal transactions to match the number of fraud transactions
under_sample_indices = np.random.choice(normal_indices, fraud_records, replace=False)
downsampled_indices = np.concatenate([fraud_indices, under_sample_indices])

# Create the downsampled dataframe
df_downsampled = df_encoded.iloc[downsampled_indices, :]

# Separate features and target in the downsampled data
X_downsampled = df_downsampled.drop(columns=['is_fraud'])
Y_downsampled = df_downsampled['is_fraud']
X_downsampled_encoded = preprocessor.transform(X_downsampled)  # Apply the same preprocessing

# Split the downsampled data into train and test sets
X_train_down, X_test_down, Y_train_down, Y_test_down = train_test_split(X_downsampled_encoded, Y_downsampled, test_size=0.30, random_state=42)

# Train a new model on downsampled data
xg_downsampled = xgb.XGBClassifier()
cmat_downsampled, pred_downsampled = RunModel(xg_downsampled, X_train_down, Y_train_down, X_test_down, Y_test_down)

# Confusion matrix for downsampled data
cm_downsampled = confusion_matrix(Y_test_down, pred_downsampled)
disp_downsampled = ConfusionMatrixDisplay(confusion_matrix=cm_downsampled)
disp_downsampled.plot()

# Model accuracy and classification report for downsampled data
print(f"Accuracy Score (Downsampled): {accuracy_score(Y_test_down, pred_downsampled)}")
print(classification_report(Y_test_down, pred_downsampled))
