import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
import pytest

from algs.apriori import a_close, apriori
from algs.rule_gen import _compare_to_mined_rules, generate_rules, generic_basis, get_classification_rules, prune_by_improvement, transitive_reduction_of_informative_basis


class TestRuleGeneration:
    def _setup(self) -> None:
        data = [  # Data-Mining context from Fast Mining of Association Rules
            [1, 3, 4],
            [2, 3, 5],
            [1, 2, 3, 5],
            [2, 5]
        ]  # Every item except for 4 is frequent
        te = TransactionEncoder()
        te_ary = te.fit_transform(data)
        transactions = pd.DataFrame(te_ary, columns=te.columns_)
        self.frequent_items = apriori(transactions, 0.5)

    def test_rule_gen_wo_pruning(self):
        self._setup()
        result = generate_rules(self.frequent_items, 0.0)
        # 4 2-itemsets + 1 3-itemset = 4*2**2-2 + 2**3-2 = 14
        assert len(result) == 14
        assert result["confidence"].min() == 2/3
        assert result["confidence"].max() == 1.0
        assert all(len(consequent) <=
                   2 for consequent in result["consequents"])
        assert all(len(antecedent) <=
                   2 for antecedent in result["antecedents"])

    def test_rule_gen_w_pruning(self):
        self._setup()
        result = generate_rules(self.frequent_items, 0.7)
        # Using the tables in the paper to calculate the confidence there should be 5
        # rules with confidence = 1 and nine with confidence = 2/3
        assert len(result) == 5
        assert all(value == 1.0 for value in result["confidence"].to_numpy())
        assert all(len(consequent) <=
                   2 for consequent in result["consequents"])
        assert all(len(antecedent) <=
                   2 for antecedent in result["antecedents"])

    def test_rule_gen_w_ignore(self):
        itemsets = pd.DataFrame([{"itemsets": (1, 2), "support": 0.2, "ignore": True}, {
            "itemsets": (3, 5), "support": 2/3, "ignore": False}, {"itemsets": (3,), "support": 2/3, "ignore": True},
            {"itemsets": (5,), "support": 0.5, "ignore": False}])
        result = generate_rules(itemsets, 0.0)
        # The 1-itemsets generate no rules and itemset {1,2} should be ignored
        assert len(result) == 2


class TestImprovement:
    def _setup(self) -> None:
        data = [  # Data Mining and Analysis:Fundamental Concepts and Algorithms Contents Table 12.1
            ["A", "B", "D", "E"],
            ["B", "C", "E"],
            ["A", "B", "D", "E"],
            ["A", "B", "C", "E"],
            ["A", "B", "C", "D", "E"],
            ["B", "C", "D"]
        ]
        te = TransactionEncoder()
        te_ary = te.fit_transform(data)
        self.transactions = pd.DataFrame(te_ary, columns=te.columns_)
        frequent_items = apriori(self.transactions, 0.5)
        self.rules = generate_rules(frequent_items, 0.6)

    def test__compare_to_mined_rules(self):
        self._setup()
        rules = get_classification_rules(self.rules, "C")
        # Only a single item in the consequent allowed
        with pytest.raises(Exception) as e:
            prune_by_improvement(self.transactions, self.rules)

        # Prune BE -> C
        rules = _compare_to_mined_rules(rules, 0.002)
        assert len(rules) == 2  # E -> C, B -> C


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

    def test_transitive_reduction_of_informative_basis(self):
        self._setup_fcs()
        gen_to_cls = {
            tuple(itemset[1]["generators"]): (tuple(itemset[1]["closed_itemsets"]), itemset[1]["support"])
            for itemset in self.fcs.iterrows()
        }
        ib = transitive_reduction_of_informative_basis(
            gen_to_cls, min_conf=3/6)

        assert len(ib) == 7

        # Table 3 in the paper shows:
        assert {"antecedents": ("C", "E"), "consequents": (
            "A", "B"), "support": 2/6, "confidence": 2/4} in ib

        assert {"antecedents": ("A",), "consequents": (
            "B", "C", "E"), "support": 2/6, "confidence": 2/3} in ib

        assert {"antecedents": ("C",), "consequents": (
            "A",), "support": 3/6, "confidence": 3/5} in ib
