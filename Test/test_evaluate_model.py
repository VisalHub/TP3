# Test ML evaluation logic using pytest.
# Write tests for evaluate_model(y_true, y_pred) that returns a dictionary with accuracy and F1 score.
# •	Accuracy = 1.0 for perfect predictions.
# •	F1 score = 0.0 when all predictions are wrong.
# •	Output contains both accuracy and f1_score keys.
import sys, os
sys.path.append(os.path.dirname(os.path.dirname(__file__)))

import unittest
from evaluation_model import evaluate_model

class TestEvaluateModel(unittest.TestCase):

    def test_perfect_predictions(self):
        y_true = [1, 0, 1, 1, 0]
        y_pred = [1, 0, 1, 1, 0]
        results = evaluate_model(y_true, y_pred)
        self.assertEqual(results['accuracy'], 1.0)
        self.assertEqual(results['f1_score'], 1.0)

    def test_all_wrong_predictions(self):
        y_true = [1, 0, 1, 1, 0]
        y_pred = [0, 1, 0, 0, 1]
        results = evaluate_model(y_true, y_pred)
        self.assertEqual(results['accuracy'], 0.0)
        self.assertEqual(results['f1_score'], 0.0)

    def test_output_keys(self):
        y_true = [1, 0, 1]
        y_pred = [1, 0, 0]
        results = evaluate_model(y_true, y_pred)
        self.assertIn('accuracy', results)
        self.assertIn('f1_score', results)

if __name__ == '__main__':
    unittest.main()
