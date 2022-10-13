from collections import defaultdict
from typing import Dict, Iterator, List, Tuple
import numpy as np
import pandas as pd
from pandas import DataFrame
from apriori import support


def ais(dataframe: DataFrame, support_threshold: float = 0.005) -> DataFrame:
    """Calculates the frequent itemsets satsifying the min support constraint, using the 
    AIS algorithm.

    Args:
        dataframe (DataFrame): All transactions. Needs to be one hot encoded.
        support_threshold (float, optional): Support threshold. Defaults to 0.005.

    Returns:
        DataFrame: Dataframe where the first column contains a list of all items in the itemset and the second
        one contains the support for that itemset. 
    """
    items = np.array(dataframe.columns)
    num_transactions = len(dataframe)
    all = __get_frequent_1_itemsets(
        items, dataframe, support_threshold)
    frequent_k_itemsets = [list(frequent_1_itemset)
                           for frequent_1_itemset in all.keys()]

    while len(frequent_k_itemsets) != 0:
        candidate_sets = defaultdict(int)
        # Iterate over potential itemsets of length k and check whether they are frequent
        for candidate_set in __generate_itemsets(frequent_k_itemsets, dataframe):
            candidate_sets[candidate_set] += 1

        frequent_k_itemsets = __get_frequent_k_itemsets(
            candidate_sets, num_transactions, support_threshold)
        all.update(frequent_k_itemsets)
        frequent_k_itemsets = [list(item)
                               for item in frequent_k_itemsets.keys()]

    # Generate dataframe from all frequent itemsets and their support
    df = pd.DataFrame(index=[i for i in range(
        len(all))], columns=['support', 'itemsets'])

    row = 0
    for itemset, support in all.items():
        df.iloc[row, 0] = support
        df.iloc[row, 1] = list(itemset)
        row += 1

    return df


def __generate_itemsets(frequent_k_itemsets: List[List[str]], transactions: DataFrame) -> Iterator[Tuple[str]]:
    """Iterates over all transactions and concatenates any item to the list of frequent k itemsets, when 
    the transaction contains the k itemset and the item has a greater lexicographic order than the last
    element in the frequent k itemset. This implies the frequent k itemsets' items to be sorted.

    Args:
        frequent_k_itemsets (List[List[str]]): Frequent itemsets of length k
        transactions (DataFrame): All transactions

    Yields:
        Iterator[Tuple[str]]: Candidate itemset of length k+1
    """
    for row in range(len(transactions)):
        transaction = list(transactions.loc[row, transactions.iloc[row]].index)
        for itemset in frequent_k_itemsets:
            if not all(item in transaction for item in itemset):
                continue
            last_element = itemset[-1]
            for item in transaction:
                if last_element < item:
                    yield tuple(itemset + [item])


def __get_frequent_1_itemsets(items: np.ndarray, transactions: DataFrame, support_threshold: float) -> Dict[Tuple[str], float]:
    """Generates all frequent 1-itemsets, satisfying the min support constraint and stores 
    them with their support in a dictionary.

    Args:
        items (np.ndarray): All items over the transactions
        transactions (DataFrame): All transactions
        support_threshold (float): Support threshold

    Returns:
        Dict[Tuple[str], float]: Frequent itemsets as tuples and their support
    """
    frequent_1_items = {}
    for item in items:
        supp = support([item], transactions)
        if supp >= support_threshold:
            frequent_1_items[(item,)] = supp

    return frequent_1_items


def __get_frequent_k_itemsets(candidate_sets: Dict[Tuple[str], int], num_transactions: int, support_threshold: float) -> Dict[Tuple[str], float]:
    """Checks whether the count for each k candidate itemset is above the min suppor threshold.

    Args:
        candidate_sets (Dict[Tuple[str], int]): Candidate sets 
        num_transactions (int): Number of transactions 
        support_threshold (float): Support threshold

    Returns:
        Dict[Tuple[str], float]: Dictionary with all frequent itemsets satisfying the min support constraint.
    """
    min_count = num_transactions * support_threshold
    return {itemset: supp / num_transactions for itemset, supp in candidate_sets.items() if supp >= min_count}
