import pandas as pd
from kafka import KafkaProducer
import json

# --------- Step 1: Load Dataset ---------
def load_data(file_path, rows=None):
    """
    Load a static dataset from a CSV file.
    If 'rows' is specified, return only the first 'rows' number of rows.
    """
    data_df = pd.read_csv(file_path)
    if rows:
        data_df = data_df.head(rows)
    return data_df

# --------- Step 2: Send Data to Kafka ---------
def send_transactions_to_kafka(producer, data_df, topic, batch_size=100):
    """
    Send rows of data to Kafka as fast as possible.
    
    Args:
        producer: KafkaProducer instance.
        data_df: DataFrame containing the transactions.
        topic: Kafka topic to send data to.
        batch_size: Number of rows to send at a time.
    """
    for i in range(0, len(data_df), batch_size):
        batch_data = data_df.iloc[i:i + batch_size].to_dict(orient='records')
        producer.send(topic, batch_data)
        print(f'Sent batch of transactions from {i} to {i + batch_size}')

# --------- Step 3: Main Execution ---------
if __name__ == "__main__":
    # Create a Kafka producer with optional compression for faster sending
    producer = KafkaProducer(
        bootstrap_servers='localhost:9092', 
        value_serializer=lambda v: json.dumps(v).encode('utf-8'),
        compression_type='gzip'  # Optional: 'gzip' or 'snappy' for compressed transmission
    )

    # Load the static dataset
    data_df = load_data('fraud_data_pandas.csv')  # Load the full dataset

    # Send data to Kafka as fast as possible
    send_transactions_to_kafka(producer, data_df, 'transaction_topic', batch_size=100)

    producer.flush()  # Ensure all messages are sent
    print("All transactions have been sent successfully.    ")