from typing import Any, Callable, Dict
from algs.apriori import a_close
from algs.fp_tree import fp_growth
from algs.hclique import hclique
from algs.quantitative import quantitative_itemsets 
from algs.rule_gen import generate_rules, minimal_non_redundant_rules
import pandas as pd


class NoMiningAlgorithmException(Exception):
    pass

class WrongArgumentException(Exception):
    pass

class NotAValidCallableException(Exception):
    pass


class Model:
    """Sets up a pipeline to transform the data a set of association rules.
    """
    def __init__(
        self, transformer: Callable, itemset_miner: Callable, rule_miner: Callable
    ) -> None:
        if not itemset_miner or not callable(itemset_miner):
            raise NoMiningAlgorithmException(
                "Need to specify an algorithm for mining frequent itemsets."
            )
        if not rule_miner or not callable(rule_miner):
            raise NoMiningAlgorithmException(
                "Need to specify an algorithm for mining rules."
            )
        self.transformer = transformer
        self.itemset_miner = itemset_miner
        self.rule_miner = rule_miner
        if self.transformer:
            self.args = {
                self.transformer: {},
                self.itemset_miner: {},
                self.rule_miner: {},
            }
        else:
            self.args = {self.itemset_miner: {}, self.rule_miner: {}}

    def set_args(self, func: Callable, args: Dict[str, Any]) -> None:
        """Associates with the function which was passed in the constructor
        all paramters.

        Args:
            func (Callable): Function that is executed at some stage of the model
            args (Dict[str, Any]): Dictionary mapping from argument names to values that 
            will be passed to the arguments having the same name. 
        """
        if self.args.get(func) == None:
            raise NotAValidCallableException("func arg must be a function that's been set in the constructor.")
        names = func.__code__.co_varnames[:func.__code__.co_argcount]
        for name in args.keys():
            if name not in names:
                raise WrongArgumentException(f"{func.__name__} does not have an argument named {name}")
        self.args[func] = args

    def run(self, data: pd.DataFrame) -> pd.DataFrame:
        """ Transforms and mines the given data, using the stored functions and arguments.

        Args:
            data (pd.DataFrame): Data to be mined

        Returns:
            pd.DataFrame: Resulting association rules
        """
        if self.transformer:
            data = self.transformer(self.args[self.transformer])
        itemsets = self.itemset_miner(data, **self.args[self.itemset_miner])
        return self.rule_miner(itemsets, **self.args[self.rule_miner])


class StandardMiner(Model):
    """Uses the fp_growth algorithm to mine frequent itemsets.
    """
    def __init__(self, transformer: Callable = None) -> None:
        super().__init__(transformer, fp_growth, generate_rules)


class HyperCliqueMiner(Model):
    """Uses the hyperclique miner for mining frequent itemsets.
    """
    def __init__(self, transformer: Callable = None) -> None:
        super().__init__(transformer, hclique, generate_rules)


class QuantitativeMiner(Model):
    """Uses the quantitative miner with dynamic interval boundaries.
    """
    def __init__(self) -> None:
        super().__init__(None, quantitative_itemsets, generate_rules)


class MinimalNonRedudantMiner(Model):
    """Determines frequent closed itemsets and then minimal non redundant rules.
    """
    def __init__(self, transformer: Callable = None) -> None:
        super().__init__(transformer, a_close, minimal_non_redundant_rules)
