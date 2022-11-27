from mlxtend.preprocessing import TransactionEncoder
import pandas as pd
import numpy as np

from algs.hclique import _prune_by_upper_bound, hclique
from algs.util import get_frequent_1_itemsets


class TestHClique:
    def _setup(self) -> None:
        data = [  # Data-Mining context from Mining Strong Affinity Association Patterns
            [1, 2],
            [1, 2],
            [1, 3, 4],
            [1, 2],
            [1, 2],
            [1, 2],
            [1, 2, 3, 4, 5],
            [1, 2],
            [1, 2],
            [2, 3, 5]
        ]
        te = TransactionEncoder()
        te_ary = te.fit_transform(data)
        self.transactions = pd.DataFrame(te_ary, columns=te.columns_)

    def test_hclique(self):
        self._setup()
        result = hclique(self.transactions, 0.55)
        hypercliques = {item[0]: item[1]
                        for item in result.to_dict('tight')['data']}
        # 5 items, which are per def. hypercliques
        assert (1,) in hypercliques
        assert (5,) in hypercliques
        assert hypercliques.get((2,)) == 0.9

        assert len(result) == 8

        # Same support level and min h-confidence constraint
        assert (1, 2) in hypercliques
        assert (3, 4) in hypercliques
        assert (3, 5) in hypercliques

        # Cross support patterns should not be in the result
        assert (1, 3) not in hypercliques
        assert (1, 4) not in hypercliques
        assert (2, 5) not in hypercliques

        # Not a cross-support pattern but h-confidence too low
        assert (4, 5) not in hypercliques

    def test_prune_by_upper_bound(self):
        self._setup()
        t = 0.55
        items = np.array(self.transactions.columns)
        all_sets = get_frequent_1_itemsets(
            items, self.transactions, 0.0)
        frequent_items = {item[0]: support for item,
                          support in all_sets.items()}

        # Remove cross-support patterns
        assert not _prune_by_upper_bound(
            t, (1, 3), frequent_items)
        assert not _prune_by_upper_bound(
            t, (2, 5), frequent_items)
        # Keep candidates, which are not cross-support patterns
        assert _prune_by_upper_bound(t, (1, 2), frequent_items)
        assert _prune_by_upper_bound(t, (4, 5), frequent_items)
