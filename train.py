import pandas as pd
import os
import xgboost as xgb
from sklearn.metrics import accuracy_score
import joblib

# Load data
def load_data(train_path, val_path):
    train_data = pd.read_csv(train_path)
    val_data = pd.read_csv(val_path)

    # Assuming the target variable is the last column
    X_train = train_data.iloc[:, :-1]
    y_train = train_data.iloc[:, -1]

    X_val = val_data.iloc[:, :-1]
    y_val = val_data.iloc[:, -1]

    return X_train, y_train, X_val, y_val

# Train model
def train_model(X_train, y_train, X_val, y_val):
    # XGBoost model initialization
    model = xgb.XGBClassifier(
        n_estimators=100,
        max_depth=5,
        learning_rate=0.1,
        objective='multi:softmax',
        num_class=7  # Adjust based on the number of classes
    )

    # Train the model
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=10)

    return model

# Save model
def save_model(model, model_dir):
    # Save the model to the provided directory
    joblib.dump(model, os.path.join(model_dir, "model.joblib"))

if __name__ == "__main__":
    # Paths to input and output data
    train_path = '/opt/ml/input/data/train/train.csv'
    val_path = '/opt/ml/input/data/val/val.csv'
    model_dir = '/opt/ml/model'

    # Load data
    X_train, y_train, X_val, y_val = load_data(train_path, val_path)

    # Train model
    model = train_model(X_train, y_train, X_val, y_val)

    # Save the model
    save_model(model, model_dir)

    # Evaluate the model
    y_pred = model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy}")
