from collections import defaultdict
from mlxtend.preprocessing import TransactionEncoder
import numpy as np
import pandas as pd
import pytest

from algs.fp_tree import FPTree, conditional_fp_tree, conditional_pattern_base, construct_fp_tree, generate_patterns_single_path, get_transformed_dataframe
from algs.util import get_frequent_1_itemsets


class TestFPTree:
    def _setup(self, min_supp: float = 0.45) -> None:
        data = [
            ["A", "B", "C"],
            ["E", "F", "C"],
            ["A", "E", "C"],
            ["E", "B"],
            ["C", "F"],
        ]
        te = TransactionEncoder()
        te_ary = te.fit_transform(data)
        self.transactions = pd.DataFrame(te_ary, columns=te.columns_)
        items = np.array(self.transactions.columns)
        frequent_items = get_frequent_1_itemsets(
            items, self.transactions, min_supp)
        self.header_table = {
            k[0]: v
            for k, v in sorted(
                frequent_items.items(), key=lambda item: item[1], reverse=True
            )
        }
        self.sorted_transactions = get_transformed_dataframe(
            self.transactions, items, self.header_table
        )

    def _setup_tree(self) -> None:
        data = [
            ["f", "a", "c", "d", "g", "i", "m", "p"],
            ["a", "b", "c", "f", "l", "m", "o"],
            ["b", "f", "h", "j", "o"],
            ["b", "c", "k", "s", "p"],
            ["a", "f", "c", "e", "l", "p", "m", "n"],
        ]
        te = TransactionEncoder()
        te_ary = te.fit_transform(data)
        transactions = pd.DataFrame(te_ary, columns=te.columns_)
        self.tree = construct_fp_tree(transactions, 0.6)

    def test_sorted_transactions(self):
        self._setup()
        new_columns = list(self.sorted_transactions.columns)
        assert new_columns == list(self.header_table.keys())
        assert len(self.sorted_transactions) == len(self.transactions)
        assert new_columns[0] == "C"
        assert new_columns[1] == "E"
        assert "A" not in new_columns
        assert "B" not in new_columns
        assert "F" not in new_columns

    def test_add_transaction(self):
        fptree = FPTree({"a": None, "z": None, "v": None})
        fptree.add_transaction(["a", "z", "v"])
        # Singe path a->z->v
        assert fptree.header_table["a"] == fptree.root.children["a"]
        assert fptree.header_table["z"] == fptree.root.children["a"].children["z"]
        assert fptree.header_table["v"] == fptree.root.children["a"].children["z"].children["v"]

        # Second path z->v
        fptree.add_transaction(["z", "v"])
        assert fptree.header_table["z"].node_link == fptree.root.children["z"]
        assert fptree.header_table["v"].node_link == fptree.root.children["z"].children["v"]

        # Prefix of path first path, thus increment count
        fptree.add_transaction(["a"])
        assert fptree.header_table["a"] == fptree.root.children["a"]
        assert fptree.root.children["a"].count == 2

        # Branch of root's a child
        fptree.add_transaction(["a", "v"])
        assert fptree.root.children["a"].count == 3
        assert fptree.header_table["v"].node_link.node_link == fptree.root.children["a"].children["v"]

    def test_add_transactions(self):
        "['C'] ['C', 'E'] ['C', 'E'] ['E'] ['C'] is the list of items in the call to add_transaction"
        self._setup()
        header_table = {k: None for k in self.header_table.keys()}
        fptree = FPTree(header_table)
        fptree.add_transactions(self.sorted_transactions)
        assert (
            fptree.header_table["C"] == fptree.root.children["C"]
        )  # only a single path to C
        assert (
            fptree.header_table["E"].node_link == fptree.root.children["E"]
        )  # Two paths to E
        assert (
            fptree.header_table["E"] == fptree.root.children["C"].children["E"]
        )  # Path C->E is the first time E is encountered

        assert fptree.root.children["C"].count == 4
        assert fptree.root.children["E"].count == 1
        assert fptree.root.children["C"].children["E"].count == 2

    @pytest.mark.parametrize("test_input, expected", [("f", 4), ("c", 4), ("a", 3), ("b", 3), ("m", 3), ("p", 3)])
    def test_get_sum_item_counts(self, test_input, expected):
        self._setup_tree()
        assert self.tree.get_sum_item_counts(test_input) == expected

    @pytest.mark.parametrize("test_input, expected", [("p", defaultdict(int, {tuple("c"): 3})),
                                                      ("m", defaultdict(
                                                          int, {("c", "f", "a"): 3})),
                                                      ("b", defaultdict(int, {})),
                                                      ("a", defaultdict(
                                                          int, {("c", "f"): 3})),
                                                      ("c", defaultdict(int, {})),
                                                      ("f", defaultdict(int, {tuple("c"): 3}))])
    def test_conditional_pattern_base(self, test_input, expected):
        self._setup_tree()
        result = conditional_pattern_base(test_input, self.tree, 3, {})
        assert result == expected

    def test_generate_patterns_single_path(self):
        result = generate_patterns_single_path(("p",), ["f", "c", "a"], 3)
        assert len(result) == 7
        assert result[("f", "c", "a", "p")] == 3
        assert result[("f", "a", "p")] == 3
        assert result[("c", "p")] == 3

    def test_conditional_fp_tree(self):
        self._setup_tree()
        header_table = {}
        pattern_base = conditional_pattern_base(
            "p", self.tree, 3, header_table)
        conditional_tree = conditional_fp_tree(pattern_base, header_table)
        assert len(conditional_tree.header_table) == 1
        assert conditional_tree.root.children["c"] == conditional_tree.header_table["c"]

        header_table = {}
        pattern_base = conditional_pattern_base(
            "m", self.tree, 3, header_table)
        conditional_tree = conditional_fp_tree(pattern_base, header_table)
        assert len(conditional_tree.header_table) == 3
        assert conditional_tree.root.children["c"] == conditional_tree.header_table["c"]
        assert conditional_tree.root.children["c"].children["f"] == conditional_tree.header_table["f"]
        assert conditional_tree.root.children["c"].children["f"].children["a"] == conditional_tree.header_table["a"]
