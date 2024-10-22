# fraud_detection_project
XGBoost Fraud Detection with Real Time Pipeline 

Project Overview:
The Real-Time Fraud Detection System is a comprehensive machine learning pipeline designed to identify fraudulent credit card transactions with high precision and recall. This system tackles one of the most challenging aspects of modern financial transactions—fraud detection—by leveraging state-of-the-art machine learning algorithms and data processing techniques.

The project is built to handle large-scale, imbalanced datasets typical in fraud detection scenarios. Using XGBoost, a highly efficient gradient boosting framework, and TensorFlow, the project integrates a hybrid approach to improve prediction accuracy. The core challenge lies in distinguishing between legitimate and fraudulent transactions within a highly skewed dataset, where fraudulent transactions make up only 0.3% of all data.

To simulate real-time fraud detection, the project incorporates Apache Kafka for streaming transactions. This allows for the continuous, real-time monitoring and classification of transactions, making the system scalable for real-world use in environments such as payment gateways or e-commerce platforms. Additionally, feature engineering plays a critical role, with preprocessing done in Apache Spark to handle 12 million+ records efficiently, extracting time-based, geographic, and transaction-specific features for enhanced fraud detection.

The system is further refined by exploring advanced model tuning techniques using Keras Tuner, ensuring optimal hyperparameters through grid search and cross-validation. Different approaches to handling class imbalance, including over/undersampling and signal integration, are used to build robust models capable of achieving high performance even on skewed data. This project serves as a scalable, real-time fraud detection framework that can be adapted to various industries dealing with high-volume transactions.

Key Features: 

Real-Time Fraud Detection: The system uses Apache Kafka to simulate and process real-time transaction data streams, enabling on-the-fly detection of fraudulent activity. Achieved a Matthews Correlation Coefficient (MCC) of 0.97 and an F1-score of 97% for real-time data pipeline predictions.

Handling Large-Scale Data: Built to efficiently process and analyze over 12 million transaction records using Apache Spark for data wrangling and preprocessing, ensuring scalability for real-world high-transaction environments.

Imbalanced Data Handling: The dataset used in this project has a typical real-world class imbalance (99.7% non-fraud vs. 0.3% fraud). The model is built to handle this imbalance effectively through advanced techniques like signal integration, over/undersampling, and weight adjustments within the machine learning pipeline.

Feature Engineering: Robust feature engineering is applied, extracting temporal (time-based), geographic (location-based), and transactional features to enhance predictive accuracy.

Machine Learning Models: Utilizes XGBoost as the primary model for fraud detection due to its performance on tabular data. The model achieves 80% precision, 75% recall, an MCC of 0.80, and an AUC-PR of 0.85 without over/undersampling techniques.

Model Stacking: A powerful ensemble model that combines XGBoost and TensorFlow, boosting predictive power to achieve 83% precision through model stacking.

Hyperparameter Tuning: Leveraged Keras Tuner to perform exhaustive hyperparameter optimization using grid search and cross-validation, resulting in highly optimized model performance.

Multiple Models for Comparison: Implemented both standard XGBoost and over/undersampled models, with the latter achieving 96% precision and recall, offering flexibility in model selection based on specific business needs.

End-to-End Pipeline: Complete fraud detection pipeline that starts from preprocessing raw data to deploying models capable of real-time predictions.

Tech Stack & Skills
Languages: Python (Pandas, NumPy), SQL (Presto)
Machine Learning: XGBoost, TensorFlow, Keras, Sklearn
Big Data: Apache Spark
Data Streaming: Apache Kafka
Visualization: Matplotlib, Seaborn
Model Tuning: Keras Tuner, Grid Search
Other Tools: Jupyter Notebook, Presto



Model Performance:

XGBoost Model (Without Over/Undersampling):
Precision: 80%
Recall: 75%
Matthews Correlation Coefficient (MCC): 0.80
AUC-PR: 0.85
Achieved these results using a highly imbalanced dataset (99.7% non-fraud, 0.3% fraud) without applying any over/undersampling techniques.

Over/Undersampled XGBoost Model:
Precision: 96%
Recall: 96%
This version of the model handles imbalanced data more effectively by using advanced sampling techniques to boost both precision and recall.

Ensemble Stacked Model (XGBoost + TensorFlow):
Precision: 83%
This stacked model leverages the strengths of both XGBoost and TensorFlow to improve the overall predictive performance, specifically targeting the minority class (fraud).

Real-Time Pipeline Predictions:
Precision: 97%
Recall: 97%
Matthews Correlation Coefficient (MCC): 0.97
Achieved these metrics with the real-time data pipeline, demonstrating the model’s ability to effectively predict fraud under real-time conditions using Apache Kafka for real-time transaction simulation and XGBoost for fraud detection.

These metrics may seem impressive but they came with lots of challenges that required extensive research from phd papers to phone calls with data scientists and or professors from universities 

Challenges Faced and Solutions:

1. Highly Imbalanced Dataset (99.7% Non-Fraud, 0.3% Fraud)

Problem: The initial dataset was extremely imbalanced, which made it difficult for the model to learn patterns related to fraudulent transactions. Models were biased towards predicting non-fraudulent transactions, lowering the model’s recall for the fraud class.

Solution:
Implemented over/undersampling techniques to balance the dataset for certain models, which boosted both precision and recall to 96%.
For models that did not use sampling techniques, I leveraged XGBoost’s scale_pos_weight parameter to account for class imbalance, achieving strong results without altering the dataset's structure.

2. Feature Engineering with Complex Data

Problem: The dataset included geographic and transactional features, which were not directly usable for model training. Raw features like latitude, longitude, and timestamps needed further transformation to be meaningful.

Solution:
Extracted advanced time-based features such as transaction hour, day of the week, and week of the year.
Engineered geographic features such as distance between transaction and merchant locations.
Used Spark for large-scale feature extraction and preprocessing on 12M+ records before converting to Pandas for model training.

3. Handling Real-Time Data Simulation

Problem: Building a system to simulate real-time transactions and fraud detection was critical for testing the model’s performance in real-world conditions.

Solution:
Integrated Apache Kafka to simulate real-time transaction streams, feeding this data directly into the XGBoost model pipeline.
Implemented a real-time fraud detection system that achieved 97% precision and recall, with an MCC of 0.97 for real-time data pipeline predictions.

4. Hyperparameter Tuning for Optimal Performance

Problem: Manually tuning hyperparameters for the XGBoost and TensorFlow models was challenging due to the large search space and complexity of both models.

Solution:
Used Keras Tuner for automatic hyperparameter tuning, leveraging cross-validation and grid search to optimize model performance.
This tuning significantly improved model metrics, increasing the F1-score and overall predictive power.

5. Ensuring Model Generalization

Problem: The model performed well on training data but needed to generalize well on unseen data to avoid overfitting.

Solution:
Utilized cross-validation during model development to ensure consistent performance across different data splits.
Regularized the model using techniques like L2 regularization and Dropout in the TensorFlow model to reduce overfitting.

Real-Time Pipeline:

This project simulates a real-time transaction streaming system using Apache Kafka. The Kafka producer streams batches of transactions in real time, which are consumed by the fraud detection system. The system classifies each transaction as either fraudulent or legitimate based on pre-trained XGBoost and TensorFlow models. This pipeline demonstrates how to handle and process large volumes of transaction data in a real-world, time-sensitive environment. Predictions from the real-time pipeline achieved an MCC of 97% and an F1-score of 97%, showcasing the model’s robust performance on real-time data.

Dataset:

The dataset used in this project contains over 12 million credit card transactions. While the dataset is publicly available, it is too large to directly import into the repository. However, a smaller, preprocessed version of the dataset, optimized for experimentation, is available in the data folder. You can also experiment with similar datasets such as the Kaggle Credit Card Fraud Detection Dataset(https://www.kaggle.com/datasets/kelvinkelue/credit-card-fraud-prediction)

Contributing
Contributions are welcome! Please open an issue or submit a pull request for any improvements or bug fixes.

License
This project is licensed under the MIT License. See the LICENSE file for details.
