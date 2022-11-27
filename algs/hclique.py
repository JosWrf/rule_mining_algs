from typing import Dict, Tuple
import numpy as np
import pandas as pd
from pandas import DataFrame
from algs.apriori import _count_transactions, _generate_itemsets_by_join, _is_candidate
from algs.hash_tree import HashTree

from algs.util import get_frequent_1_itemsets


def hclique(dataframe: DataFrame, hconf_threshold: float = 0.5, support_threshold: float = 0.0) -> DataFrame:
    """Implements the hclique-miner algorithm from the 'Mining Strong Affinity Association Patterns in 
    Data Sets with Skewed Support Distribution' paper in order to determine all hyperclique patterns.
    (The cross-support property of the h-confidence is not used to prune cross-support itemsets before they 
    are generated.)

    Args:
        dataframe (DataFrame): Transactional database
        hconf_threshold (float, optional): Minimum h-confidence threshold. Defaults to 0.5.
        support_threshold (float, optional): Minimum support threshold. Defaults to 0.0.

    Returns:
        DataFrame: All hyperclique patterns, satisfying min support and min h-confidence constraints, with 
        their support values.
    """
    items = np.array(dataframe.columns)
    all_sets = get_frequent_1_itemsets(
        items, dataframe, support_threshold)
    frequent_items = {item[0]: support for item, support in all_sets.items()}
    frequent_k_itemsets = [
        frequent_1_itemset for frequent_1_itemset in all_sets.keys()]
    k = 1

    while len(frequent_k_itemsets) != 0:
        hash_tree = HashTree()

        for candidate_set in _generate_itemsets_by_join(frequent_k_itemsets, k):
            # Prune wrt. antimonotone property of support/h-conf and cross-support upper bound of h-conf
            if _is_candidate(frequent_k_itemsets, candidate_set) and _prune_by_upper_bound(hconf_threshold, candidate_set, frequent_items):
                hash_tree.add_itemset(candidate_set)

        _count_transactions(dataframe, hash_tree, k)

        frequent_k_itemsets = hash_tree.get_frequent_itemsets(
            support_threshold, len(dataframe)
        )

        frequent_k_itemsets = _prune_by_hconf_threshold(
            frequent_k_itemsets, frequent_items, hconf_threshold)

        all_sets.update(frequent_k_itemsets)
        frequent_k_itemsets = sorted(frequent_k_itemsets.keys())
        k += 1

        # Generate dataframe from all frequent itemsets and their support
        df = pd.DataFrame(
            all_sets.items(),
            index=[i for i in range(len(all_sets))],
            columns=["itemsets", "support"],
        )

        return df


def _prune_by_upper_bound(hconf_threshold: float, pattern: Tuple[str], items: Dict[str, float]) -> bool:
    """Prunes the candidate pattern, based on the cross-support, which is an upper bound on the 
    h-confidence.

    Args:
        hconf_threshold (float): Minimum h-confidence threshold
        pattern (Tuple[str]): Candidate pattern 
        items (Dict[str, float]): Frequent 1 items

    Returns:
        bool: Returns true when the item is not a cross-support pattern else false.
    """
    min_item = 1
    max_item = -1

    for item in pattern:
        support = items.get(item)
        min_item = min(support, min_item)
        max_item = max(support, max_item)

    return min_item / max_item >= hconf_threshold


def _prune_by_hconf_threshold(frequent_k_itemsets: Dict[Tuple[str], float], items: Dict[str, float], hconf_threshold: float) -> Dict[Tuple[str], float]:
    """Removes any patterns whose h-confidence is smaller than the given h-confidence threshold.

    Args:
        frequent_k_itemsets (Dict[Tuple[str], float]): Frequent k itemsets, satisfying the min support constraint
        items (Dict[str, float]): Frequent 1 itemsets
        hconf_threshold (float): h-confidence threshold

    Returns:
        Dict[Tuple[str], float]: Patterns satisfying the threshold.
    """
    result = {}
    for itemset, supp in frequent_k_itemsets.items():
        max_item = -1
        for item in itemset:
            max_item = max(max_item, items.get(item))
        if supp / max_item >= hconf_threshold:
            result[itemset] = supp

    return result
