import unittest
import pandas as pd

from src.data import load_chia


class DataTestCase(unittest.TestCase):
    def setUp(self):
        """Load Chia dataset as a Pandas dataframe"""
        self.df = load_chia()

    def test_load_chia_returns_dataframe(self):
        """Tests if load_chia returns a Pandas dataframe"""
        self.assertIsInstance(self.df, pd.DataFrame)

    def test_chia_number_of_rows(self):
        """Tests if load_chia returns a dataframe with 2000 rows"""
        self.assertEqual(self.df.shape[0], 2000)

    def test_chia_number_of_columns(self):
        """Tests if load_chia returns a dataframe with 12 columns"""
        self.assertEqual(self.df.shape[1], 12)


if __name__ == "__main__":
    unittest.main()
