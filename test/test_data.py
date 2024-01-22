import unittest
import pandas as pd

from src.data import load_chia, load_fb


class DataTestCase(unittest.TestCase):
    def setUp(self):
        """Load Chia dataset as a Pandas dataframe"""
        self.df_chia = load_chia()
        self.df_fb = load_fb()

    def test_load_chia_returns_dataframe(self):
        """Tests if load_chia returns a Pandas dataframe"""
        self.assertIsInstance(self.df_chia, pd.DataFrame)

    def test_chia_number_of_rows(self):
        """Tests if load_chia returns a dataframe with 2000 rows"""
        self.assertEqual(self.df_chia.shape[0], 2000)

    def test_chia_number_of_columns(self):
        """Tests if load_chia returns a dataframe with 12 columns"""
        self.assertEqual(self.df_chia.shape[1], 12)

    def test_fb_returns_dataframe(self):
        """Tests if load_fb returns a Pandas dataframe"""
        self.assertIsInstance(self.df_fb, pd.DataFrame)

    def test_fb_number_of_rows(self):
        """Tests if load_fb returns a dataframe with 49903 rows"""
        self.assertEqual(self.df_fb.shape[0], 49903)

    def test_fb_number_of_columns(self):
        """Tests if load_fb returns a dataframe with 17 columns"""
        self.assertEqual(self.df_fb.shape[1], 17)


if __name__ == "__main__":
    unittest.main()
