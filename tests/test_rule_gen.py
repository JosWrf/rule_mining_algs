import pandas as pd
from mlxtend.preprocessing import TransactionEncoder

from algs.apriori import a_close
from algs.rule_gen import generic_basis


class TestMinimalNonRedundantRules:
    def _setup_fcs(self, min_support: float = 2/6) -> None:
        data = [  # Data-Mining context from Mining non-redundant ARs paper
            ["A", "C", "D"],
            ["B", "C", "E"],
            ["A", "B", "C", "E"],
            ["B", "E"],
            ["A", "B", "C", "E"],
            ["B", "C", "E"],
        ]
        te = TransactionEncoder()
        te_ary = te.fit_transform(data)
        transactions = pd.DataFrame(te_ary, columns=te.columns_)
        self.fcs = a_close(transactions, min_support)

    def test_generic_basis(self):
        self._setup_fcs()
        gen_to_cls = {
            tuple(itemset[1]["generators"]): (tuple(itemset[1]["closed_itemsets"]), itemset[1]["support"])
            for itemset in self.fcs.iterrows()
        }
        gb = generic_basis(gen_to_cls)

        assert len(gb) == 7  # Table 2 in the paper shows solutions
        assert {"antecedents": ("A",), "consequents": (
            "C",), "support": 3/6, "confidence": 1} in gb

        assert {"antecedents": ("A", "E"), "consequents": (
            "B", "C"), "support": 2/6, "confidence": 1} in gb

        assert {"antecedents": ("C", "E"), "consequents": (
            "B",), "support": 4/6, "confidence": 1} in gb
