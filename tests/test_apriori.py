from mlxtend.preprocessing import TransactionEncoder
import numpy as np
import pandas as pd

from algs.apriori import (
    _generate_itemsets_by_join,
    _remove_same_closure_as_subset,
    a_close,
    apriori,
    closure,
)

from algs.util import get_frequent_1_itemsets


class TestAPriori:
    def _setup(self) -> None:
        data = [  # Data-Mining context from APriori paper
            ["1", "3", "4"],
            ["2", "3", "5"],
            ["1", "2", "3", "5"],
            ["2", "5"],
        ]
        te = TransactionEncoder()
        te_ary = te.fit_transform(data)
        self.transactions = pd.DataFrame(te_ary, columns=te.columns_)

    def test_apriori(self):
        self._setup()
        result = apriori(self.transactions, 0.5)
        assert len(result) == 9
        frequent_itemsets = {
            result.loc[row, "itemsets"]: result.loc[row, "support"]
            for row in range(len(result))
        }
        assert frequent_itemsets[("2", "3", "5")] == 0.5
        assert frequent_itemsets[("2", "3")] == 0.5
        assert frequent_itemsets[("1",)] == 0.5
        assert frequent_itemsets[("5",)] == 0.75

    def test_generate_candidates_by_join(self):
        old_candidates = [("1",), ("2",), ("3",), ("5",)]
        k = 1
        result = [
            candidate for candidate in _generate_itemsets_by_join(old_candidates, k)
        ]
        assert len(result) == 6
        assert ("1", "5") in result
        assert ("2", "3") in result


class TestAClose:
    def _setup(self) -> None:
        data = [  # Data-Mining context from A-Close paper
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
        assert set(result["closed_itemsets"]) == set(
            [("A", "C"), ("B", "E"), ("C",), ("A", "B", "C", "E"), ("B", "C", "E")]
        )

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
        assert set(result.keys()) == set(
            [("A", "C"), ("B", "E"), ("C",), ("A", "B", "C", "E"), ("B", "C", "E")]
        )

    def test_closure_idempotency(self):
        self._setup()

        result = closure(
            self.transactions,
            [
                {
                    ("A", "C"): 0.6,
                    ("B", "E"): 0.8,
                    ("C",): 0.8,
                    ("A", "B", "C", "E"): 0.4,
                    ("B", "C", "E"): 0.6,
                }
            ],
        )
        assert set(result.keys()) == set(
            [("A", "C"), ("B", "E"), ("C",), ("A", "B", "C", "E"), ("B", "C", "E")]
        )

    def test_closed_itemsets(self):
        self._setup()
        result = a_close(self.transactions, 0.4)
        closed_itemsets = {
            result.loc[row, "closed_itemsets"]: result.loc[row, "support"]
            for row in range(len(result))
        }
        closed = closure(self.transactions, [closed_itemsets])
        assert set(closed.keys()) == set(
            [("A", "C"), ("B", "E"), ("C",), ("A", "B", "C", "E"), ("B", "C", "E")]
        )

    def test_remove_same_closure_as_subset(self):
        self._setup()
        items = np.array(self.transactions.columns)
        old_generators = get_frequent_1_itemsets(items, self.transactions, 0.4)
        candidate_generators = {
            ("A", "B"): 0.4,
            ("A", "C"): 0.6,
            ("A", "E"): 0.4,
            ("B", "C"): 0.6,
            ("B", "E"): 0.8,
            ("C", "E"): 0.6,
        }
        result, found_unclosed = _remove_same_closure_as_subset(
            candidate_generators, old_generators
        )
        assert found_unclosed
        assert len(result) == 4
