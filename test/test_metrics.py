import unittest
from src.metrics import jaccard_score


class MetricsTestCase(unittest.TestCase):
    def test_jaccard_score_empty_sets(self):
        """Tests if jaccard_score returns 0. for empty sets"""
        self.assertEqual(jaccard_score(set(), set()), 0.)

    def test_jaccard_score_identical_sets(self):
        """Tests if jaccard_score returns 1. for identical sets"""
        self.assertEqual(jaccard_score({1, 2, 3}, {1, 2, 3}), 1.)

    def test_jaccard_score_disjoint_sets(self):
        """Tests if jaccard_score returns 0. for disjoint sets"""
        self.assertEqual(jaccard_score({1, 2, 3}, {4, 5, 6}), 0.)

    def test_jaccard_score_left_empty(self):
        """Tests if jaccard_score returns 0. for empty left set"""
        self.assertEqual(jaccard_score(set(), {1, 2, 3}), 0.)

    def test_jaccard_score_right_empty(self):
        """Tests if jaccard_score returns 0. for empty right set"""
        self.assertEqual(jaccard_score({1, 2, 3}, set()), 0.)

    def test_jaccard_score_strict(self):
        """Tests if jaccard_score returns correct value for strict mode"""
        self.assertEqual(jaccard_score({1, 2, 3}, {2, 3, 4}, mode="strict"), 0.5)

    def test_jaccard_score_relaxed(self):
        """Tests if jaccard_score returns correct value for relaxed mode"""
        self.assertEqual(jaccard_score({0, 1, 2, 3}, {2, 3, 4, 5, 6}, mode="relaxed"), 0.5)

    def test_jaccard_score_left(self):
        """Tests if jaccard_score returns correct value for left mode"""
        self.assertEqual(jaccard_score({0, 1, 2, 3}, {2, 3, 4}, mode="left"), 0.5)

    def test_jaccard_score_right(self):
        """Tests if jaccard_score returns correct value for right mode"""
        self.assertEqual(jaccard_score({0, 1, 2}, {1, 2, 3, 4}, mode="right"), 0.5)


if __name__ == '__main__':
    unittest.main()
