import boto3
import pandas as pd
from sklearn.model_selection import train_test_split

# S3 Client Setup
s3 = boto3.client('s3')

def download_data(bucket_name, object_key, local_path):
    # Download data from S3
    s3.download_file(bucket_name, object_key, local_path)

def preprocess_data(input_path, output_train, output_val, output_test):
    # Load data
    data = pd.read_csv(input_path)

    # Replace categorical values
    data[16] = data[16].replace({
        'SEKER': 0,
        'BARBUNYA': 1,
        'BOMBAY': 2,
        'CALI': 3,
        'HOROZ': 4,
        'SIRA': 5,
        'DERMASON': 6
    })

    # Shuffle and split data
    data = data.sample(frac=1).reset_index(drop=True)
    train_data, temp_data = train_test_split(data, test_size=0.3, random_state=42)
    val_data, test_data = train_test_split(temp_data, test_size=0.5, random_state=42)

    # Save processed datasets
    train_data.to_csv(output_train, index=False)
    val_data.to_csv(output_val, index=False)
    test_data.to_csv(output_test, index=False)

if __name__ == "__main__":
    bucket_name = 'drybean-csv'
    object_key = 'data/drybean.csv'
    local_data_path = '/opt/ml/processing/input/data.csv'
    
    # Define paths to save preprocessed data
    output_train = '/opt/ml/processing/output/train.csv'
    output_val = '/opt/ml/processing/output/val.csv'
    output_test = '/opt/ml/processing/output/test.csv'

    # Download and preprocess data
    download_data(bucket_name, object_key, local_data_path)
    preprocess_data(local_data_path, output_train, output_val, output_test)
