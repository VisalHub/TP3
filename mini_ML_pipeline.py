# Create and test a mini ML pipeline with three functions: 
# 1. load_data() - loads and returns a DataFrame. 
# 2. train_model() - trains a simple model (e.g., logistic regression). 
# 3. evaluate_model() - returns accuracy. 
# • Data loads correctly (non-empty, correct columns). 
# • Model trains without error. 
# • Accuracy is between 0 and 1.
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.datasets import load_iris
import numpy as np

def load_data():
    # Load the iris dataset as an example
    iris = load_iris()
    df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
    df['target'] = iris.target
    return df
def train_model(X_train, y_train):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    return model
def evaluate_model(y_true, y_pred):
    accuracy = accuracy_score(y_true, y_pred)
    return {'accuracy': accuracy}

if __name__ == "__main__":
    # Load data
    data = load_data()
    print("Data loaded successfully with shape:", data.shape)

    # Split data
    X = data.drop('target', axis=1)
    y = data['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train model
    model = train_model(X_train, y_train)
    print("Model trained successfully.")

    # Evaluate model
    y_pred = model.predict(X_test)
    results = evaluate_model(y_test, y_pred)
    print("Model evaluation results:", results)
    assert 0.0 <= results['accuracy'] <= 1.0, "Accuracy is out of bounds!"
    print("Accuracy is within valid range (0 to 1).")
