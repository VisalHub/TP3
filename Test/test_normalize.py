# Apply Test-Driven Development (TDD).
# Write tests first for a function normalize_column(df, column) that scales values between 0 and 1.
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pandas as pd
import unittest
from normalize_column import normalize_column

class TestNormalizeColumn(unittest.TestCase):

    def test_normalized_values_within_bounds(self):
        data = {'A': [10, 20, 30], 'B': [40, 50, 60]}
        df = pd.DataFrame(data)
        normalized_df = normalize_column(df.copy(), 'A')
        self.assertTrue(((normalized_df['A'] >= 0) & (normalized_df['A'] <= 1)).all())

    def test_output_length_matches_input(self):
        data = {'A': [5, 15, 25, 35], 'B': [45, 55, 65, 75]}
        df = pd.DataFrame(data)
        normalized_df = normalize_column(df.copy(), 'A')
        self.assertEqual(len(normalized_df), len(df))

    def test_invalid_column_raises_keyerror(self):
        data = {'A': [1, 2, 3], 'B': [4, 5, 6]}
        df = pd.DataFrame(data)
        with self.assertRaises(KeyError):
            normalize_column(df.copy(), 'C')

if __name__ == '__main__':
    unittest.main()



