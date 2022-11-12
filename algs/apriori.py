import pandas as pd
import numpy as np
from typing import Dict, Iterator, List, Tuple
from pandas import DataFrame
from algs.util import get_frequent_1_itemsets

from algs.hash_tree import HashTree


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
    all_sets = get_frequent_1_itemsets(items, dataframe, support_threshold)
    frequent_k_itemsets = [
        frequent_1_itemset for frequent_1_itemset in all_sets.keys()]
    k = 1

    while len(frequent_k_itemsets) != 0:
        # Iterate over potential itemsets of length k and check whether they are frequent
        hash_tree = HashTree()

        for candidate_set in _generate_itemsets_by_join(frequent_k_itemsets, k):
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


def _generate_itemsets_by_join(
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
        for j in range(i + 1, len(old_itemsets)):
            skip = False
            for l in range(k - 1):
                if old_itemsets[i][l] != old_itemsets[j][l]:
                    skip = True
                    break

            if not skip and old_itemsets[i][k - 1] < old_itemsets[j][k - 1]:
                yield old_itemsets[i] + (old_itemsets[j][k - 1],)


def __is_candidate(old_itemsets: List[Tuple[str]], candidate_set: Tuple[str]) -> bool:
    """Checks whether there's any subset contained in the candidate_set, that isn't
    contained within the old_itemsets. If that is the case the candidate set can not
    be a frequent itemset and False is returned.

    Args:
        old_itemsets (List[Tuple[str]]): List of itemsets of length k
        candidate_set (Tuple[str]): Candidate itemset with length k+1

    Returns:
        bool: True if all k-1 element subsets of candidate_set are contained within old_itemsets.
    """
    # Joining two 1 frequent itemsets, every subset must be frequent
    if len(candidate_set) == 2:
        return True

    for i in range(len(candidate_set)):
        if not candidate_set[0:i] + candidate_set[i + 1:] in old_itemsets:
            return False

    return True


def __count_transactions(transactions: DataFrame, tree: HashTree, k: int) -> None:
    """Iterates over all transactions and uses them to traverse the hash tree. If a
    leaf is encountered all itemsets at that leaf are compared against the transaction
    and their count is incremented by 1.

    Args:
        transactions (DataFrame): All transactions
        tree (HashTree): HashTree containing candidate itemsets
        k (int): Length of candidate itemsets
    """
    for idx, row in transactions.iterrows():
        transaction = list(transactions.loc[idx:, list(row)])
        tree.transaction_counting(transaction, 0, k + 1, dict())


def a_close(dataframe: DataFrame, support_threshold: float = 0.005) -> DataFrame:
    """Implementation of the a-close algorithm according to 'Discovering frequent closed itemsets
    for association rules'.

    Args:
        dataframe (DataFrame): All transactions, one-hot encoded, columns are lexicographically sorted.
        support_threshold (float, optional): Minimum support threshold. Defaults to 0.005.

    Returns:
        DataFrame: Generators, their frequent closed itemsets and support
    """
    # Calculate frequent 1-generators
    items = np.array(dataframe.columns)
    generators = [get_frequent_1_itemsets(items, dataframe, support_threshold)]
    current_generators = [
        frequent_1_itemset for frequent_1_itemset in generators[0].keys()
    ]
    closed_level = 0
    k = 1

    while len(current_generators) != 0:
        # Build (i+1)-generators by combining frequent (i)-generators and count support
        hash_tree = HashTree()

        for candidate_set in _generate_itemsets_by_join(current_generators, k):
            if __is_candidate(current_generators, candidate_set):
                hash_tree.add_itemset(candidate_set)

        __count_transactions(dataframe, hash_tree, k)

        current_generators = hash_tree.get_frequent_itemsets(
            support_threshold, len(dataframe)
        )

        # Remove generators having the same support as one of their i-subsets
        current_generators, found = _remove_same_closure_as_subset(
            current_generators, generators[k-1])
        closed_level = k if found and closed_level == 0 else closed_level

        generators.append(current_generators)
        current_generators = sorted(current_generators.keys())
        k += 1

    # Calculate closure for all generators at index >= level
    generators_and_closures = {}
    for k_generators in generators[:closed_level-1]:
        for k_generator, supp in k_generators.items():
            generators_and_closures[k_generator] = (k_generator, supp)
    if closed_level > 0:
        generators_and_closures.update(
            closure(dataframe, generators[closed_level-1:-1]))

    # Generate dataframe from all generators their closed frequent itemsets and support
    df = pd.DataFrame(
        index=[i for i in range(len(generators_and_closures))],
        columns=["generators", "closed_itemsets", "support"],
    )

    i = 0
    for generator, closed in generators_and_closures.items():
        df.loc[i, "generators"] = generator
        df.loc[i, "closed_itemsets"] = closed[0]
        df.loc[i, "support"] = closed[1]
        i += 1

    return df


def _remove_same_closure_as_subset(
    current_generators: Dict[Tuple[str], float], all_generators: Dict[Tuple[str], float]
) -> Tuple[Dict[Tuple[str], float], bool]:
    """Prunes all (i+1)-generators, that have a subset i-generator with the same support.
    This implies their closure is the same and thus the (i+1)-generator is redundant.

    Args:
        current_generators (Dict[Tuple[str], float]): (i+1)-generators
        all_generators (Dict[Tuple[str], float]): i-generators

    Returns:
        Tuple[Dict[Tuple[str], float], bool]: Dictonary with all redundant generators removed and
        a flag indicating, whether a generator happened to be redundant.
    """
    same_closure = False
    pruned_generators = {}

    for candidate_set, supp in current_generators.items():
        valid = True
        for i in range(len(candidate_set)):
            i_generator = candidate_set[0:i] + candidate_set[i + 1:]
            if all_generators[i_generator] == supp:
                same_closure = True
                valid = False
                break

        if valid:
            pruned_generators[candidate_set] = supp

    return pruned_generators, same_closure


def closure(transactions: DataFrame, unclosed_generators: List[Dict[Tuple[str], float]]) -> Dict[Tuple[str], float]:
    """Calculates the galois closure operator h. It receives a list of potentially unclosed generators and
    updates the closure for each generator by building the intersection with f(o) where o is a transaction.

    Args:
        transactions (DataFrame): All transactions
        unclosed_generators (List[Dict[Tuple[str], float]]): Generators for i >= level, having at least one unclosed generator

    Returns:
        Dict[Tuple[str], Tuple[str,float]]: The generator and their closure for all itemsets in unclosed_generators.
    """
    fc = {generator: [set(), supp]
          for i_generators in unclosed_generators for generator, supp in i_generators.items()}

    for idx, row in transactions.iterrows():
        transaction = list(transactions.loc[idx:, list(row)])

        for p in fc.keys():
            if set(p).issubset(transaction):
                if len(fc[p][0]) == 0:
                    fc[p][0] = set(transaction)
                else:
                    fc[p][0] = fc[p][0].intersection(set(transaction))

    return {generator: (tuple(sorted(closure[0])), closure[1]) for generator, closure in fc.items()}
