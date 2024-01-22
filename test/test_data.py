import unittest
import pandas as pd

from src.data import load_chia, load_fb, train_test_dev_split


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

    def test_train_test_dev_split(self):
        """Tests if train_test_dev_split returns a dictionary with 3 keys"""
        self.assertEqual(len(train_test_dev_split(self.df_chia)), 3)

    def test_train_test_dev_split_keys(self):
        """Tests if train_test_dev_split returns a dictionary with train, test, and dev keys"""
        self.assertIn("train", train_test_dev_split(self.df_chia))
        self.assertIn("test", train_test_dev_split(self.df_chia))
        self.assertIn("dev", train_test_dev_split(self.df_chia))

    def test_train_test_dev_split_wrong_ratios(self):
        """Tests if train_test_dev_split raises AssertionError when sum of ratios is not 100"""
        with self.assertRaises(AssertionError):
            train_test_dev_split(self.df_chia, ratio=(70, 20, 11))

    def test_train_test_dev_split_sizes_of_splits(self):
        """Tests if train_test_dev_split returns a dictionary with train, test, and dev splits of correct sizes"""
        splits = train_test_dev_split(self.df_chia, ratio=(70, 20, 10))
        self.assertEqual(splits["train"].shape[0], 1400)
        self.assertEqual(splits["test"].shape[0], 400)
        self.assertEqual(splits["dev"].shape[0], 200)


if __name__ == "__main__":
    unittest.main()
