import pandas as pd
import numpy as np
from typing import Dict, Iterator, List, Tuple
from pandas import DataFrame

from hash_tree import HashTree


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


def apriori(dataframe: DataFrame, support_threshold: float = 0.005) -> DataFrame:
    """Calculate all frequent itemsets for the given transactions and support
    threshold.

    Args:
        dataframe (DataFrame): All transactions stored in the dataframe. Needs to be one hot encoded.
        support_threshold (float, optional): Min threshold used to prune candidate itemsets

    Returns:
        DataFrame: Dataframe where the first column contains a list of all items in the itemset and the second
        one contains the support for that itemset.
    """
    items = np.array(dataframe.columns)
    all_sets = __get_frequent_1_itemsets(items, dataframe, support_threshold)
    frequent_k_itemsets = [frequent_1_itemset for frequent_1_itemset in all_sets.keys()]
    k = 1

    while len(frequent_k_itemsets) != 0:
        # Iterate over potential itemsets of length k and check whether they are frequent
        hash_tree = HashTree()

        for candidate_set in __generate_itemsets_by_join(frequent_k_itemsets, k):
            if __is_candidate(frequent_k_itemsets, candidate_set):
                hash_tree.add_itemset(candidate_set)

        __count_transactions(dataframe, hash_tree, k)

        frequent_k_itemsets = hash_tree.get_frequent_itemsets(
            support_threshold, len(dataframe)
        )

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


def __generate_itemsets_by_join(
    old_itemsets: List[Tuple[str]], k: int
) -> Iterator[Tuple[str]]:
    """Joins frequent k-1 itemsets to generate k itemsets.
    It assumes the frequent k-1 itemsets are lexicographically ordered .

    Args:
        old_itemsets (List[Tule[str]]): List of itemsets of length k-1
        k (int): The number of items that must match to join two frequent k-1 itemsets

    Yields:
        Iterator[Tuple[str]]: A candidate k itemset
    """
    for i in range(len(old_itemsets)):
        for j in range(i+1, len(old_itemsets)):
            skip = False
            for l in range(k - 1):
                if old_itemsets[i][l] != old_itemsets[j][l]:
                    skip = True
                    break

            if not skip and old_itemsets[i][k - 1] < old_itemsets[j][k - 1]:
                yield old_itemsets[i] + (old_itemsets[j][k - 1],)


def __is_candidate(old_itemsets: List[Tuple[str]], candidate_set: np.ndarray) -> bool:
    """Checks whether there's any subset contained in the candidate_set, that isn't
    contained within the old_itemsets. If that is the case the candidate set can not
    be a frequent itemset and False is returned.

    Args:
        old_itemsets (List[Tuple[str]]): List of itemsets of length k
        candidate_set (np.ndarray): Candidate itemset with length k+1

    Returns:
        bool: True if all k-1 element subsets of candidate_set are contained within old_itemsets.
    """
    # Joining two 1 frequent itemsets, every subset must be frequent
    if len(candidate_set) == 2:
        return True

    for i in range(len(candidate_set)):
        if not candidate_set[0:i] + candidate_set[i + 1 :] in old_itemsets:
            return False

    return True


def __get_frequent_1_itemsets(
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


def __count_transactions(transactions: DataFrame, tree: HashTree, k: int) -> None:
    """Iterates over all transactions and uses them to traverse the hash tree. If a 
    leaf is encountered all itemsets at that leaf are compared against the transaction 
    and their count is incremented by 1.

    Args:
        transactions (DataFrame): All transactions
        tree (HashTree): HashTree containing candidate itemsets
        k (int): Length of candidate itemsets
    """
    for row in range(len(transactions)):
        transaction = list(transactions.loc[row, list(transactions.loc[row])].index)
        tree.transaction_counting(transaction, 0, k + 1, dict())
