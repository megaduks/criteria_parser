import unittest
from src.metrics import *


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
            entity_match_score(
                [["a b c d", "e f g h"], ["i j k", "l m n"]], ["a b", "e f"]
            ),
            0.0,
        )

    def test_entity_match_ents_predicted_with_none(self):
        """Tests if entity_match_score returns 0. for one empty list"""
        self.assertEqual(
            entity_match_score([["a", "b c"], ["d", "e f"]], ["None"]), 0.0
        )

    def test_entity_match_ents_true_empty_ents_predicted_with_none(self):
        """Tests if entity_match_score returns 0. for one empty list"""
        self.assertEqual(entity_match_score([], ["None"]), 0.0)

    def test_llm_none_cleaner_single_None(self):
        """Tests if llm_none_cleaner returns an empty list for a list containing a single 'None'"""
        self.assertEqual(llm_none_cleaner(["None"]), [])

    def test_llm_none_cleaner_nonempty_list(self):
        """Tests if llm_none_cleaner returns the original list for a nonempty list"""
        self.assertEqual(llm_none_cleaner(["a", "b", "c"]), ["a", "b", "c"])

    def test_llm_none_cleaner_single_None_with_period(self):
        """Tests if llm_none_cleaner returns an empty list for a list containing a single 'None' with a period"""
        self.assertEqual(llm_none_cleaner(["None."]), [])

    def test_llm_none_cleaner_single_none_lowercase(self):
        """Tests if llm_none_cleaner returns an empty list for a list containing a single 'none'"""
        self.assertEqual(llm_none_cleaner(["none"]), [])

    def test_llm_none_cleaner_two_Nones(self):
        """Tests if llm_none_cleaner returns the original list for a list containing two 'None's"""
        self.assertEqual(llm_none_cleaner(["None", "None"]), ["None", "None"])

    def test_str_to_BIO_entities_simple_entity(self):
        """Tests if str_to_BIO_entities returns the correct BIO entities for a simple entity"""
        self.assertEqual(
            str_to_BIO_entities("a b c d", "ENT", ["b"]), ["O", "B-ENT", "O", "O"]
        )

    def test_str_to_BIO_multi_token_entity(self):
        """Tests if str_to_BIO_entities returns the correct BIO entities for a multi-token entity"""
        self.assertEqual(
            str_to_BIO_entities("aaa bbb ccc ddd", "ENT", ["bbb ccc"]),
            ["O", "B-ENT", "I-ENT", "O"],
        )

    def test_str_to_BIO_empty_entities(self):
        """Tests if str_to_BIO_entities returns the correct BIO entities for an empty list of entities"""
        self.assertEqual(
            str_to_BIO_entities("a b c d", "ENT", []), ["O", "O", "O", "O"]
        )

    def test_str_to_BIO_multiple_multi_token_entities(self):
        """Tests if str_to_BIO_entities returns the correct BIO entities for multiple multi-token entities"""
        self.assertEqual(
            str_to_BIO_entities(
                "aaa bbb ccc ddd eee fff ggg hhh", "ENT", ["bbb ccc", "eee fff"]
            ),
            ["O", "B-ENT", "I-ENT", "O", "B-ENT", "I-ENT", "O", "O"],
        )

    def test_str_to_BIO_nonexistent_entity(self):
        """Tests if str_to_BIO_entities returns the correct BIO entities for a nonexistent entity"""
        self.assertEqual(
            str_to_BIO_entities("a b c d", "ENT", ["e"]), ["O", "O", "O", "O"]
        )

    def test_str_to_BIO_entity_in_wrong_order_of_tokens(self):
        """Tests if str_to_BIO_entities returns the correct BIO entities for an entity with wrong order"""
        self.assertEqual(
            str_to_BIO_entities("a b c d", "ENT", ["c b"]), ["O", "O", "O", "O"]
        )

    def test_str_to_BIO_entity_with_punctuation(self):
        """Tests if str_to_BIO_entities returns the correct BIO entities for an entity with punctuation"""
        self.assertEqual(
            str_to_BIO_entities("a b c, d.", "ENT", ["b c"]),
            ["O", "B-ENT", "I-ENT", "O", "O", "O"],
        )

    def test_merge_BIO_entities_empty(self):
        """Tests if merge_BIO_entities returns an empty list for an empty list of BIO entities"""
        self.assertEqual(merge_BIO_entities([]), [])

    def test_merge_BIO_entities_single_O(self):
        """Tests if merge_BIO_entities returns the original list for a list containing a single 'O'"""
        self.assertEqual(merge_BIO_entities([["O"]]), ["O"])

    def test_merge_BIO_entities_two_lists_different_entities(self):
        """Tests if merge_BIO_entities returns the original list for a list containing two different entities"""
        self.assertEqual(
            merge_BIO_entities([["B-X", "I-X", "O", "O"], ["O", "O", "B-Y", "I-Y"]]),
            ["B-X", "I-X", "B-Y", "I-Y"],
        )

    def test_merge_BIO_entities_two_lists_overlapping_entities(self):
        """Tests if merge_BIO_entities returns the original list for a list containing two overlapping entities"""
        self.assertEqual(
            merge_BIO_entities([["B-X", "I-X", "O", "O"], ["B-Y", "I-Y", "O", "O"]]),
            ["B-Y", "I-Y", "O", "O"],
        )

    def test_merge_BIO_entities_three_input_lists(self):
        """Tests if merge_BIO_entities returns the original list for a list containing three input lists"""
        self.assertEqual(
            merge_BIO_entities(
                [
                    ["B-X", "I-X", "O", "O"],
                    ["B-Y", "I-Y", "O", "O"],
                    ["O", "O", "B-Z", "I-Z"],
                ]
            ),
            ["B-Y", "I-Y", "B-Z", "I-Z"],
        )


if __name__ == "__main__":
    unittest.main()
