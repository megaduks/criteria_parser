import unittest
import pandas as pd

from src.data import load_chia, load_fb, train_test_val_split, get_chia_annotations


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

    def test_fb_returns_dict(self):
        """Tests if load_fb returns a dictionary"""
        self.assertIsInstance(self.df_fb, dict)

    def test_fb_keys(self):
        """Tests if load_fb returns a dictionary with 3 keys"""
        self.assertEqual(len(self.df_fb), 3)
        self.assertIn("train", self.df_fb)
        self.assertIn("test", self.df_fb)
        self.assertIn("val", self.df_fb)

    def test_fb_num_rows(self):
        """Tests if load_fb returns a dictionary with 3 dataframes of correct sizes"""
        self.assertEqual(self.df_fb["train"].shape[0], 1243)
        self.assertEqual(self.df_fb["test"].shape[0], 10116)
        self.assertEqual(self.df_fb["val"].shape[0], 376)

    def test_train_test_val_split(self):
        """Tests if train_test_val_split returns a dictionary with 3 keys"""
        self.assertEqual(len(train_test_val_split(self.df_chia)), 3)

    def test_train_test_val_split_keys(self):
        """Tests if train_test_val_split returns a dictionary with train, test, and val keys"""
        self.assertIn("train", train_test_val_split(self.df_chia))
        self.assertIn("test", train_test_val_split(self.df_chia))
        self.assertIn("val", train_test_val_split(self.df_chia))

    def test_train_test_val_split_wrong_ratios(self):
        """Tests if train_test_val_split raises AssertionError when sum of ratios is not 100"""
        with self.assertRaises(AssertionError):
            train_test_val_split(self.df_chia, ratio=(70, 20, 11))

    def test_train_test_val_split_sizes_of_splits(self):
        """Tests if train_test_val_split returns a dictionary with train, test, and val splits of correct sizes"""
        splits = train_test_val_split(self.df_chia, ratio=(70, 20, 10))
        self.assertEqual(splits["train"].shape[0], 1400)
        self.assertEqual(splits["test"].shape[0], 400)
        self.assertEqual(splits["val"].shape[0], 200)

    def test_get_chia_annotations_returns_list_of_tuples(self):
        """Tests if get_chia_annotations returns a list of tuples"""
        self.assertIsInstance(get_chia_annotations("drugs"), list)
        self.assertIsInstance(get_chia_annotations("drugs")[0], tuple)

    def test_get_chia_annotations_returns_limited_num_rows(self):
        """Tests if get_chia_annotations returns a list of tuples of the correct length"""
        self.assertEqual(len(get_chia_annotations("drugs", n=10)), 10)

    def test_get_chia_annotations_raises_error_for_wrong_entity(self):
        """Tests if get_chia_annotations raises ValueError for wrong entity"""
        with self.assertRaises(AssertionError):
            get_chia_annotations("wrong_entity")


if __name__ == "__main__":
    unittest.main()
