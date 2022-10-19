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

