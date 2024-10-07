from kafka import KafkaConsumer, KafkaProducer
import json
import pandas as pd
import pickle

# Load XGBoost model and preprocess pipeline
with open('fraud_detection_pipeline.pkl', 'rb') as f:
    pipeline_dict = pickle.load(f)

xgb_model = pipeline_dict['model']
label_encoder_merchant = pipeline_dict['label_encoder_merchant']
label_encoder_city = pipeline_dict['label_encoder_city']
label_encoder_age_group = pipeline_dict['label_encoder_age_group']

# Kafka consumer for transactions
consumer = KafkaConsumer(
    'transaction_topic',
    bootstrap_servers='localhost:9092',
    auto_offset_reset='earliest',
    enable_auto_commit=True,
    group_id='fraud-detection-group',
    value_deserializer=lambda x: json.loads(x.decode('utf-8'))
)

# Kafka producer for sending predictions
producer = KafkaProducer(bootstrap_servers='localhost:9092', value_serializer=lambda v: json.dumps(v).encode('utf-8'))

# To store results
results = []

# Function to preprocess data for XGBoost
def preprocess_data(transaction):
    df = pd.DataFrame([transaction])

    # Use the loaded label encoders
    df['merchant'] = label_encoder_merchant.transform([transaction['merchant']])[0]
    df['city'] = label_encoder_city.transform([transaction['city']])[0]
    df['age_group'] = label_encoder_age_group.transform([transaction['age_group']])[0]
    
    # Drop unnecessary columns
    df = df.drop(columns=['is_fraud', 'trans_timestamp'])

    return df

# Consume transactions, preprocess, and predict
for message in consumer:
    batch_transactions = message.value
    print(f"Received batch of transactions: {batch_transactions}")

    # Process each transaction in the batch
    for transaction in batch_transactions:
        try:
            # Preprocess the transaction for prediction
            transaction_df = preprocess_data(transaction)
            
            # Predict using the XGBoost model
            prediction = xgb_model.predict(transaction_df)
            
            # Print prediction result
            result = {"transaction": transaction, "prediction": int(prediction[0])}
            print(f"Prediction: {result}")
            
            # Append result to the list
            results.append(result)
            
            # Optionally send the prediction to another Kafka topic
            producer.send('prediction_topic', result)
            print(f'Sent prediction: {result}')
        except Exception as e:
            print(f"Error processing transaction: {e}")

    # After processing the batch, save the results to a CSV file
    results_df = pd.DataFrame(results)
    results_df.to_csv('predictions.csv', index=False)
    print("Predictions saved to predictions.csv")