import unittest
from src.metrics import jaccard_score, entity_coverage_score, entity_match_score


class MetricsTestCase(unittest.TestCase):
    def test_jaccard_score_empty_sets(self):
        """Tests if jaccard_score returns 0. for empty sets"""
        self.assertEqual(jaccard_score(set(), set()), 0.0)

    def test_jaccard_score_identical_sets(self):
        """Tests if jaccard_score returns 1. for identical sets"""
        self.assertEqual(jaccard_score({1, 2, 3}, {1, 2, 3}), 1.0)

    def test_jaccard_score_disjoint_sets(self):
        """Tests if jaccard_score returns 0. for disjoint sets"""
        self.assertEqual(jaccard_score({1, 2, 3}, {4, 5, 6}), 0.0)

    def test_jaccard_score_left_empty(self):
        """Tests if jaccard_score returns 0. for empty left set"""
        self.assertEqual(jaccard_score(set(), {1, 2, 3}), 0.0)

    def test_jaccard_score_right_empty(self):
        """Tests if jaccard_score returns 0. for empty right set"""
        self.assertEqual(jaccard_score({1, 2, 3}, set()), 0.0)

    def test_jaccard_score_strict(self):
        """Tests if jaccard_score returns correct value for strict mode"""
        self.assertEqual(jaccard_score({1, 2, 3}, {2, 3, 4}, mode="strict"), 0.5)

    def test_jaccard_score_relaxed(self):
        """Tests if jaccard_score returns correct value for relaxed mode"""
        self.assertEqual(
            jaccard_score({0, 1, 2, 3}, {2, 3, 4, 5, 6}, mode="relaxed"), 0.5
        )

    def test_jaccard_score_left(self):
        """Tests if jaccard_score returns correct value for left mode"""
        self.assertEqual(jaccard_score({0, 1, 2, 3}, {2, 3, 4}, mode="left"), 0.5)

    def test_jaccard_score_right(self):
        """Tests if jaccard_score returns correct value for right mode"""
        self.assertEqual(jaccard_score({0, 1, 2}, {1, 2, 3, 4}, mode="right"), 0.5)

    def test_entity_coverage_score_empty(self):
        """Tests if entity_coverage_score returns 0. for empty lists"""
        self.assertEqual(entity_coverage_score([], []), 0.0)

    def test_entity_coverage_score_identical_lists(self):
        """Tests if entity_coverage_score returns 1. for identical lists"""
        self.assertEqual(entity_coverage_score(["a", "b", "c"], ["a", "b", "c"]), 1.0)

    def test_entity_coverage_score_disjoint_lists(self):
        """Tests if entity_coverage_score returns 0. for disjoint lists"""
        self.assertEqual(entity_coverage_score(["a", "b", "c"], ["d", "e", "f"]), 0.0)

    def test_entity_coverage_score_left_empty(self):
        """Tests if entity_coverage_score returns 0. for empty left list"""
        self.assertEqual(entity_coverage_score([], ["a", "b", "c"]), 0.0)

    def test_entity_coverage_score_right_empty(self):
        """Tests if entity_coverage_score returns 0. for empty right list"""
        self.assertEqual(entity_coverage_score(["a", "b", "c"], []), 0.0)

    def test_entity_coverage_score_strict(self):
        """Tests if entity_coverage_score returns correct value for strict mode"""
        self.assertEqual(
            entity_coverage_score(
                ["a", "b", "c", "d"], ["b", "c", "d", "e"], jaccard_mode="strict"
            ),
            0.75,
        )

    def test_entity_coverage_score_relaxed(self):
        """Tests if entity_coverage_score returns correct value for relaxed mode"""
        self.assertEqual(
            entity_coverage_score(
                ["a", "b", "c", "d"], ["b", "c", "d", "e"], jaccard_mode="relaxed"
            ),
            0.75,
        )

    def test_entity_coverage_score_left(self):
        """Tests if entity_coverage_score returns correct value for left mode"""
        self.assertEqual(
            entity_coverage_score(
                ["b", "c"], ["b", "c", "d", "e"], jaccard_mode="left"
            ),
            1.0,
        )

    def test_entity_coverage_score_right(self):
        """Tests if entity_coverage_score returns correct value for right mode"""
        self.assertEqual(
            entity_coverage_score(
                ["b", "c", "d", "e"], ["b", "c"], jaccard_mode="right"
            ),
            0.5,
        )

    def test_entity_coverage_score_partial_matching_entities(self):
        """Tests if entity_coverage_score returns correct value for partial matching entities"""
        self.assertEqual(
            entity_coverage_score(["a b c d", "e f g h"], ["a b", "e f"]), 0.5
        )

    def test_entity_coverage_score_partial_matching_with_unmatched_entities(self):
        """Tests if entity_coverage_score returns correct value for partial matching entities"""
        self.assertEqual(
            entity_coverage_score(["a b c d", "e f g h", "i j k", "l m n"], ["a b"]),
            0.125,
        )

    def test_entity_coverage_score_left_argument_is_not_a_list(self):
        """Tests if entity_coverage_score raises TypeError when left argument is not a list"""
        with self.assertRaises(TypeError):
            entity_coverage_score("a b c d", ["a b"])

    def test_entity_coverage_score_right_argument_is_not_a_list(self):
        """Tests if entity_coverage_score raises TypeError when right argument is not a list"""
        with self.assertRaises(TypeError):
            entity_coverage_score(["a b c d"], "a b")

    def test_entity_match_emplty_lists(self):
        """Tests if entity_match_score returns 0. for empty lists"""
        self.assertEqual(entity_match_score([], []), 0.0)

    def test_entity_match_equal_lists(self):
        """Tests if entity_match_score returns 1. for equal lists"""
        self.assertEqual(
            entity_match_score(
                [["a", "b c"], ["d", "e f"]], [["a", "b c"], ["d", "e f"]]
            ),
            1.0,
        )

    def test_entity_match_one_prediction_empty(self):
        """Tests if entity_match_score returns 0. for one empty list"""
        self.assertEqual(entity_match_score([["a", "b c"], ["d", "e f"]], []), 0.0)

    def test_entity_match_partial_matching_entities(self):
        """Tests if entity_match_score returns correct value for partial matching entities"""
        self.assertEqual(
            entity_match_score([["a b c d", "e f g h"], ["i j k", "l m n"]], ["a b", "e f"]),
            0.0,
        )

    def test_entity_match_ents_predicted_with_none(self):
        """Tests if entity_match_score returns 0. for one empty list"""
        self.assertEqual(entity_match_score([["a", "b c"], ["d", "e f"]], ['None']), 0.0)

    def test_entity_match_ents_true_empty_ents_predicted_with_none(self):
        """Tests if entity_match_score returns 0. for one empty list"""
        self.assertEqual(entity_match_score([], ["None"]), 0.0)


if __name__ == "__main__":
    unittest.main()
