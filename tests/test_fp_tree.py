from mlxtend.preprocessing import TransactionEncoder
import numpy as np
import pandas as pd

from algs.fp_tree import FPTree, get_transformed_dataframe
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
        frequent_items = get_frequent_1_itemsets(items, self.transactions, min_supp)
        self.header_table = {
            k[0]: v
            for k, v in sorted(
                frequent_items.items(), key=lambda item: item[1], reverse=True
            )
        }
        self.sorted_transactions = get_transformed_dataframe(
            self.transactions, items, self.header_table
        )

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
        fptree.add_transaction(["a","z","v"])
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
