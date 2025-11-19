import pandas as pd
import unittest 
from clean_data import clean_data

class TestCleanData(unittest.TestCase):

    def test_remove_duplicates(self):
        data = {'A': [1, 2, 2, 3], 'B': [4, 5, 5, 6]}
        df = pd.DataFrame(data)
        cleaned_df = clean_data(df)
        self.assertEqual(len(cleaned_df), 3)
        self.assertFalse(cleaned_df.duplicated().any())

    def test_drop_nulls(self):
        data = {'A': [1, None, 3], 'B': [4, 5, None]}
        df = pd.DataFrame(data)
        cleaned_df = clean_data(df)
        self.assertEqual(len(cleaned_df), 1)
        self.assertFalse(cleaned_df.isnull().any().any())

    def test_row_count_decrease(self):
        data = {'A': [1, 2, 2, None], 'B': [4, None, 5, 6]}
        df = pd.DataFrame(data)
        initial_row_count = len(df)
        cleaned_df = clean_data(df)
        self.assertLess(len(cleaned_df), initial_row_count)
        
if __name__ == '__main__':
    unittest.main()