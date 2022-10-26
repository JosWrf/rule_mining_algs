from typing import List, Tuple
from algs.hash_tree import HashTree


class TestHashTree:
    def _setup_tree(self, max_size: int) -> None:
        self.tree = HashTree(max_size=max_size)

    def _setup_tree_with_items(self, max_size: int, itemsets: List[Tuple[str]]) -> None:
        self._setup_tree(max_size)
        for tple in itemsets:
            self.tree.add_itemset(tple)

    def test_number_items(self):
        tples = [
            ("Bier", "Donut"),
            ("Wasser", "Weizen"),
            ("Mehl", "Bier"),
            ("Bier", "Pizza"),
        ]
        self._setup_tree_with_items(2, tples)
        assert self.tree.number_items() == 4

        self._setup_tree_with_items(4, tples)
        assert self.tree.number_items() == 4

        self._setup_tree_with_items(1, [("Hefe", "Nudeln")])
        assert self.tree.number_items() == 1

    def test_insert(self):
        self._setup_tree(2)
        assert self.tree.number_items() == 0
        self.tree.add_itemset(("B", "C"))
        assert self.tree.number_items() == 1
        assert self.tree.itemsets[("B", "C")] == 0
        assert len(self.tree.children) == 0

        self.tree.add_itemset(("E", "F"))
        assert self.tree.number_items() == 2
        assert self.tree.itemsets[("E", "F")] == 0
        assert self.tree.leaf
        assert len(self.tree.children) == 0

        self.tree.add_itemset(("B", "E"))
        assert self.tree.number_items() == 3
        assert not self.tree.leaf
        assert len(self.tree.children) == 2
        assert (
            len(self.tree.children[self.tree.hash_func("B", self.tree.depth)].itemsets)
            == 2
        )
        assert (
            len(self.tree.children[self.tree.hash_func("E", self.tree.depth)].itemsets)
            == 1
        )

        print("Miep")
        self.tree.add_itemset(("B", "G"))
        assert len(self.tree.children) == 2
        assert self.tree.number_items() == 4
        assert not self.tree.leaf
        b_child = self.tree.children[self.tree.hash_func("B", self.tree.depth)]
        e_child = self.tree.children[self.tree.hash_func("E", self.tree.depth)]
        assert b_child.depth == 1
        assert b_child.depth == 1
        assert e_child.leaf
        assert not b_child.leaf
        assert len(b_child.children) == 3
        assert len(b_child.itemsets) == 0

    def test_hash_func(self):
        tree = HashTree()
        assert tree.hash_func(("B", "E"), 0) == 3
        assert tree.hash_func(("B", "E"), 1) == 6
        assert tree.hash_func(("A"), 0) == 2
        assert tree.hash_func(("Bier",), 0) == 1

    def test_get_frequent_itemsets(self):
        tpls = [("A", "F"), ("A", "G"), ("B", "K"), ("N", "O"), ("S", "P"), ("A", "K"), ("A", "B")]
        self._setup_tree_with_items(2, tpls)
        transactions = [["A", "G", "B"], ["A", "S", "G", "F", "P"]]
        for transaction in transactions:
            self.tree.transaction_counting(transaction, 0, 2, {})
        frequent_itemsets = self.tree.get_frequent_itemsets(0.5, len(transactions))
        assert len(frequent_itemsets) == 4
        frequent_itemsets = self.tree.get_frequent_itemsets(1, len(transactions))
        assert len(frequent_itemsets) == 1

    def test_simple_transaction_counting(self):
        tpls = [("A", "K"), ("A", "G"), ("B", "K"), ("S", "P")]
        self._setup_tree_with_items(3, tpls)
        assert self.tree.number_items() == len(tpls)
        assert len(self.tree.children) == 3
        itemsets = self.tree.get_all_itemsets()
        for tpl in tpls:
            assert itemsets[tpl] == 0
        transaction = sorted(["F","A","S","K"])
        self.tree.transaction_counting(transaction, 0, 2, {})
        itemsets = self.tree.get_all_itemsets()
        assert sum(itemsets.values()) == 1

    def test_triple_transaction_counting(self):
        tpls = [("A", "C", "E"), ("A", "C", "F"), ("B", "M", "X")]
        self._setup_tree_with_items(1, tpls)
        assert self.tree.number_items() == len(tpls)
        assert len(self.tree.children) == 2
        itemsets = self.tree.get_all_itemsets()
        for tpl in tpls:
            assert itemsets[tpl] == 0
        transaction = ["A", "B", "C", "E", "F", "X"]
        self.tree.transaction_counting(transaction, 0, 3, {})
        itemsets = self.tree.get_all_itemsets()
        assert sum(itemsets.values()) == 2



    def test_transaction_counting(self):
        tpls = [("A", "F"), ("A", "G"), ("B", "K"), ("N", "O"), ("S", "P"), ("A", "K"), ("A", "B")]
        self._setup_tree_with_items(2, tpls)
        assert self.tree.number_items() == len(tpls)
        assert len(self.tree.children) == 4

        itemsets = self.tree.get_all_itemsets()
        for tpl in tpls:
            assert itemsets[tpl] == 0

        transaction = ["A", "G", "B"]
        self.tree.transaction_counting(transaction, 0, 2, {})
        itemsets = self.tree.get_all_itemsets()
        assert sum(itemsets.values()) == 2

        transaction = ["A", "S", "G", "F", "P"]
        self.tree.transaction_counting(transaction, 0, 2, {})
        itemsets = self.tree.get_all_itemsets()
        # Old count is 2 and new count would be 3
        assert sum(itemsets.values()) == 5

        transaction = ["P"]
        self.tree.transaction_counting(transaction, 0, 2, {})
        itemsets = self.tree.get_all_itemsets()
        assert sum(itemsets.values()) == 5
