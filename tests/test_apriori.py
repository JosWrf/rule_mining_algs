from mlxtend.preprocessing import TransactionEncoder
import numpy as np
import pandas as pd
from algs.apriori import a_close, closure

from algs.util import get_frequent_1_itemsets


class TestAClose:
    def _setup(self) -> None:
        data = [  # Data-Mining paper from A-Close paper
            ["A", "C", "D"],
            ["B", "C", "E"],
            ["A", "B", "C", "E"],
            ["B", "E"],
            ["A", "B", "C", "E"],
        ]
        te = TransactionEncoder()
        te_ary = te.fit_transform(data)
        self.transactions = pd.DataFrame(te_ary, columns=te.columns_)

    def test_get_frequent_1_itemsets(self):
        self._setup()
        items = np.array(self.transactions.columns)
        frequent_items = get_frequent_1_itemsets(items, self.transactions, 0.4)
        assert len(frequent_items) == 4
        assert list(sorted(frequent_items.keys())) == [("A",), ("B",), ("C",), ("E",)]

    def test_a_close(self):
        self._setup()
        result = a_close(self.transactions, 0.4)
        assert len(result) == 5
        assert set(result['closed_itemsets']) == set([("A", "C"), ("B", "E"), ("C",), ("A","B","C","E"), ("B","C","E")])

    def test_closure(self):
        self._setup()
        items = np.array(self.transactions.columns)
        frequent_items = get_frequent_1_itemsets(items, self.transactions, 0.4)
        result = closure(self.transactions, [frequent_items])

        assert set(result.keys()) == set([("A", "C"), ("B", "E"), ("C",)])
        result = closure(
            self.transactions,
            [frequent_items]
            + [{("A", "B"): 0.4, ("A", "E"): 0.4, ("B", "C"): 0.6, ("C", "E"): 0.6}],
        )
        assert set(result.keys()) == set([("A", "C"), ("B", "E"), ("C",), ("A","B","C","E"), ("B","C","E")])

