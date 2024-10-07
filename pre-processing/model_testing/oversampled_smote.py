import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
from sklearn.metrics import (precision_recall_curve, auc, accuracy_score,
                             precision_score, recall_score, confusion_matrix,
                             classification_report)

from imblearn.over_sampling import SMOTE

# Importing the data
df = pd.read_csv("fraud_data_pandas.csv")

# Normalizing the Amount column
sc = StandardScaler()
df['NormalizedAmount'] = sc.fit_transform(df['total_amount'].values.reshape(-1, 1))

# Dropping unnecessary columns
df.drop(['trans_timestamp', 'total_amount', 'merchant'], axis=1, inplace=True)

# Encoding categorical features
label_encoder = LabelEncoder()
df['age_group'] = label_encoder.fit_transform(df['age_group'])
df['city'] = label_encoder.fit_transform(df['city'])

# Checking the data distribution before resampling
prct_classes_before = df['is_fraud'].value_counts(normalize=True)
prct_classes_before.plot(kind='bar')
plt.ylabel('Frequency')
plt.title('Fraud Class Distribution Before Resampling')
plt.show()

# Preparing data for training
y = df['is_fraud']
X = df.drop(['is_fraud'], axis=1)

# Create the SMOTE instance with a sampling strategy to equalize classes
smote = SMOTE(sampling_strategy='auto')  # Automatically balances classes

# Fit and resample
X_resample, y_resample = smote.fit_resample(X, y)
print("Resampled class distribution:")
print(y_resample.value_counts())

# Plotting the distribution after resampling
prct_classes_after = y_resample.value_counts(normalize=True)
prct_classes_after.plot(kind='bar')
plt.ylabel('Frequency')
plt.title('Fraud Class Distribution After Resampling (Balanced)')
plt.show()

# Splitting the resampled data into train/test sets
X_train, X_test, y_train, y_test = train_test_split(X_resample, y_resample, test_size=0.20, random_state=25)

# Training the model
model_resample = XGBClassifier()
model_resample.fit(X_train, y_train)
predictions = model_resample.predict(X_test)

# Plotting the precision-recall curve
precision, recall, _ = precision_recall_curve(y_test, predictions)
plt.figure(figsize=(10, 6))
plt.plot(recall, precision, linestyle='-', color='blue', label='Precision-Recall Curve')
plt.xlabel('Recall')
plt.ylabel('Precision')
plt.title('Precision-Recall Curve')
plt.xlim([0, 1])
plt.ylim([0, 1])
plt.legend()
plt.show()

# Calculating metrics
accuracy = accuracy_score(y_test, predictions)
precision = precision_score(y_test, predictions)
recall = recall_score(y_test, predictions)
print("Model Accuracy : ", accuracy)
print('Precision on testing set:', precision)
print('Recall on testing set:', recall)
print(classification_report(y_test, predictions))
