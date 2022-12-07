import math
from typing import Dict, List, Tuple
import numpy as np

from pandas import DataFrame


def support(subset: List[str], data_df: DataFrame) -> float:
    """Calculates the support for a given itemset over all transactions.

    Args:
        subset (List[str]): List containing a candidate itemset
        data_df (DataFrame): Contains all itemsets

    Returns:
        float: Support for the itemset
    """
    numberTransactions = len(data_df)
    itemset_count = data_df.loc[:, subset].all(axis=1).sum()

    return itemset_count / numberTransactions

def get_frequent_1_itemsets(
    items: np.ndarray, transactions: DataFrame, support_threshold: float
) -> Dict[Tuple[str], float]:
    """Calculates all frequent 1 itemsets and returns them aswell as their support.

    Args:
        items (np.ndarray): Numpy array of all items
        transactions (DataFrame): The set of all transactions
        support_threshold (float): Support threshold

    Returns:
        Dict[Tuple[str], float]: Frequent 1 itemsets and their support
    """
    frequent_1_item_sets = {}
    for item in items:
        supp = support([item], transactions)
        if support_threshold <= supp:
            frequent_1_item_sets[(item,)] = supp

    return frequent_1_item_sets

def lift(supp_antecedent: float, supp_consequent: float, supp_union: float) -> float:
    return supp_union / (supp_antecedent * supp_consequent)

def cosine(supp_antecedent: float, supp_consequent: float, supp_union: float) -> float:
    return supp_union / math.sqrt(supp_antecedent * supp_consequent) 

def independent_cosine(supp_antecedent: float, supp_consequent: float) -> float:
    return math.sqrt(supp_consequent * supp_antecedent)

def imbalance_ratio(supp_antecedent: float, supp_consequent: float, supp_union: float) -> float:
    return abs(supp_antecedent - supp_consequent) / (supp_antecedent + supp_consequent - supp_union)

def kulczynski(supp_antecedent: float, supp_consequent: float, supp_union: float) -> float:
    return 0.5*(confidence(supp_antecedent, supp_union) + confidence(supp_consequent, supp_union))

def confidence(supp_antecedent: float, supp_union: float) -> float:
    return supp_union / supp_antecedent

def conviction(supp_antecedent: float, supp_consequent: float, supp_union: float) -> float:
    return (1-supp_consequent) / (1-confidence(supp_antecedent, supp_union))
