# Create and test a mini ML pipeline with three functions: 
# 1. load_data() - loads and returns a DataFrame. 
# 2. train_model() - trains a simple model (e.g., logistic regression). 
# 3. evaluate_model() - returns accuracy. 
# • Data loads correctly (non-empty, correct columns). 
# • Model trains without error. 
# • Accuracy is between 0 and 1.

import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import unittest
from mini_ML_pipeline import load_data, train_model, evaluate_model 

class TestMiniMLPipeline(unittest.TestCase):

    def test_load_data(self):
        data = load_data()
        self.assertIsNotNone(data)
        self.assertFalse(data.empty)
        self.assertIn('target', data.columns)

    def test_train_model(self):
        data = load_data()
        X = data.drop('target', axis=1)
        y = data['target']
        from sklearn.model_selection import train_test_split
        X_train, _, y_train, _ = train_test_split(X, y, test_size=0.2, random_state=42)
        try:
            model = train_model(X_train, y_train)
            self.assertIsNotNone(model)
        except Exception as e:
            self.fail(f"Model training failed with exception: {e}")

    def test_evaluate_model(self):
        data = load_data()
        X = data.drop('target', axis=1)
        y = data['target']
        from sklearn.model_selection import train_test_split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        model = train_model(X_train, y_train)
        y_pred = model.predict(X_test)
        results = evaluate_model(y_test, y_pred)
        self.assertIn('accuracy', results)
        self.assertGreaterEqual(results['accuracy'], 0.0)
        self.assertLessEqual(results['accuracy'], 1.0)
if __name__ == '__main__':
    unittest.main()
