import pandas as pd
import numpy as np
from typing import Iterator, List, Tuple
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
    frequent_1_itemsets, set_support = __get_frequent_1_itemsets(
        items, dataframe, support_threshold)
    frequent_k_itemsets = [[frequent_1_itemset]
                           for frequent_1_itemset in frequent_1_itemsets]
    all = frequent_k_itemsets

    while len(frequent_k_itemsets) != 0:
        temp = []
        # Iterate over potential itemsets of length k and check whether they are frequent
        for candidate_set in __generate_itemsets(frequent_k_itemsets, frequent_1_itemsets):
            if __is_candidate(frequent_k_itemsets, candidate_set):
                supp = support(candidate_set, dataframe)
                if supp >= support_threshold:
                    temp.append(candidate_set)
                    set_support.append(supp)

        frequent_k_itemsets = temp
        all.extend(temp)

    # Generate dataframe from all frequent itemsets and their support
    df = pd.DataFrame(index=[i for i in range(
        len(all))], columns=['support', 'itemsets'])

    for i in range(0, len(all)):
        df.iloc[i, 0] = set_support[i]
        df.iloc[i, 1] = all[i]

    return df


def __generate_itemsets(old_itemsets: List[List[str]], frequent_1_itemsets: np.ndarray) -> Iterator[List[str]]:
    """Concatenates frequent 1 itemsets to all frequent k-1 itemsets to generate candidate
    k itemsets. It assumes the frequent k-1 itemsets are ordered and only appends frequent 1 itemsets
    to the end, when they are lexicographically greater than the last element in the old_itemset to be expanded.

    Args:
        old_itemsets (List[List[str]]): List of itemsets of length k-1
        frequent_1_itemsets (np.ndarray): The frequent 1 itemsets

    Yields:
        Iterator[List[str]]: A candidate k itemset 
    """
    for old_itemset in old_itemsets:
        last_item = old_itemset[-1]
        for frequent_1_item in frequent_1_itemsets:
            if frequent_1_item > last_item:
                yield old_itemset + [frequent_1_item]


def __is_candidate(old_itemsets: List[List[str]], candidate_set: np.ndarray) -> bool:
    """Checks whether there's any subset contained in the candidate_set, that isn't
    contained within the old_itemsets. If that is the case the candidate set can not 
    be a frequent itemset and False is returned.

    Args:
        old_itemsets (List[List[str]]): List of itemsets of length k
        candidate_set (np.ndarray): Candidate itemset with length k+1 

    Returns:
        bool: True if all k-1 element subsets of candidate_set are contained within old_itemsets.
    """
    # Joining two 1 frequent itemsets, every subset must be frequent
    if len(candidate_set) == 2:
        return True

    for i in range(len(candidate_set)):
        if not candidate_set[0:i] + candidate_set[i+1:] in old_itemsets:
            return False

    return True


def __get_frequent_1_itemsets(items: np.ndarray, transactions: DataFrame, support_threshold: float) -> Tuple[np.ndarray, List[float]]:
    """Calculates all frequent 1 itemsets and returns them aswell as their support.

    Args:
        items (np.ndarray): Numpy array of all items
        transactions (DataFrame): The set of all transactions
        support_threshold (float): Support threshold

    Returns:
        Tuple[np.ndarray, List[float]]: Frequent 1 itemsets and their support
    """
    frequent_1_item_sets = []
    supports = []
    for item in items:
        supp = support([item], transactions)
        if support_threshold <= supp:
            frequent_1_item_sets.append(item)
            supports.append(supp)

    return np.array(frequent_1_item_sets), supports
