import unittest
import pandas as pd

from src.data import load_chia


class DataTestCase(unittest.TestCase):
    def test_load_chia_returns_dataframe(self):
        """Tests if load_chia returns a Pandas dataframe"""
        self.assertIsInstance(load_chia(), pd.DataFrame)


if __name__ == "__main__":
    unittest.main()
