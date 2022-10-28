from typing import DefaultDict, Dict, List, Tuple
import numpy as np

from pandas import DataFrame
from collections import defaultdict

from util import get_frequent_1_itemsets


class FPNode:
    """Node used in a fp tree.
    """

    def __init__(self, item: str, parent: "FPNode", count: int = 1) -> None:
        self.node_link = None
        self.parent = parent
        self.item = item
        self.count = count
        self.children = {}


class FPTree:
    """Class used for fp trees and conditional fp trees.
    """

    def __init__(self, header_table: Dict[str, None]) -> None:
        self.root = FPNode(None, None)
        self.header_table = header_table

    def add_transaction(self, transaction: List[str], node_count: int = 1) -> None:
        """Encodes a sorted (e.g. by support) transaction to a path in the fp tree.

        Args:
            transaction (List[str]): Sorted transaction
            node_count (int, optional): Should only be set when construction conditional fp trees. 
            Defaults to 1.
        """

        def __add_transaction(depth: 0, node: FPNode) -> None:
            if depth == len(transaction):
                return

            item_name = transaction[depth]
            child = node.children.get(item_name)
            if child != None:
                child.count += node_count

            else:
                child = FPNode(item_name, node, node_count)
                node.children[item_name] = child
                self.__set_node_link(item_name, child)

            __add_transaction(depth + 1, child)

        __add_transaction(0, self.root)

    def __set_node_link(self, item_name: str, node: FPNode) -> None:
        """Set the node_link for a node or add an entry to the header table.

        Args:
            item_name (str): Name of the item 
            node (FPNode): Node to link to
        """
        next_node = self.header_table.get(item_name)
        if next_node == None:
            self.header_table[item_name] = node

        else:
            while next_node != None:
                previous_node = next_node
                next_node = next_node.node_link

            previous_node.node_link = node

    def add_transactions(self, transactions: DataFrame) -> None:
        """Iterates over the list of sorted (e.g. support) transactions
        and calls add_transaction.

        Args:
            transactions (DataFrame): All transactions to build the fp tree from.
        """
        for row in range(len(transactions)):
            transaction = list(
                transactions.loc[row, list(transactions.loc[row])].index)
            self.add_transaction(transaction)

    def get_sum_item_counts(self, item: str) -> int:
        """Given a item in the header list, sum all counts of nodes 
        that can be reached via node_links starting from the header link.

        Args:
            item (str): Item in the header list

        Returns:
            int: Total count for the respective item
        """
        header_item = self.header_table[item]
        count_sum = 0

        while header_item != None:
            count_sum += header_item.count
            header_item = header_item.node_link

        return count_sum


def fp_growth(transactions: DataFrame, min_support: float = 0.05) -> DataFrame:
    """Uses the fp_growth method described by Han et al. to calculate all frequent itemsets
    satisfying the minimum support constraint.

    Args:
        transactions (DataFrame): One-Hot encoded dataframe containing all transactions.
        min_support (float, optional): Minimum support threshold. Defaults to 0.05.

    Returns:
        DataFrame: Dataframe where the first column contains a list of all items in the itemset and the second
        one contains the support for that itemset.
    """
    fptree = construct_fp_tree(
        transactions, min_support)
    min_supp = int(min_support*len(transactions))
    itemsets = fp_tree_growth(fptree, min_supp)
    fp_itemsets = DataFrame(itemsets.items(), index=[i for i in range(
        len(itemsets))], columns=['itemsets', 'support'])
    fp_itemsets["support"] = fp_itemsets["support"] / len(transactions)
    fp_itemsets = fp_itemsets[fp_itemsets['support'] > min_support]

    return fp_itemsets


def get_transformed_dataframe(old_df: DataFrame, all_items: np.ndarray, frequent_items: Dict[str, float]) -> DataFrame:
    """Removes all infrequent itemsets from the transactions. It sorts the transactions in 
    descending order of support.

    Args:
        old_df (DataFrame): All transactions.
        all_items (np.ndarray): Items contained in the transactions.
        frequent_items (Dict[str, float]): All frequent items in the transactions.

    Returns:
        DataFrame: New dataframe with all infrequent items removed and remaining items sorted in
        descending order of support.
    """
    drop_columns = [item for item in all_items if not frequent_items.get(item)]
    return old_df.drop(drop_columns, inplace=False, axis=1)[frequent_items.keys()]


def construct_fp_tree(transactions: DataFrame, min_support: float) -> FPTree:
    """Constructs a fp_tree from the given transactions and minimum support
    threshold.

    Args:
        transactions (DataFrame): All transactions.
        min_support (float): Minimum support threshold.

    Returns:
        FPTree: fp tree containing all frequent itemset information.
    """
    # Get frequent items and sort transactions
    items = np.array(transactions.columns)
    frequent_items = get_frequent_1_itemsets(items, transactions, min_support)
    frequent_items = {k[0]: v for k, v in sorted(
        frequent_items.items(), key=lambda item: item[1], reverse=True)}
    sorted_transactions = get_transformed_dataframe(
        transactions, items, frequent_items)
    # Build header table for node links and construct FP tree
    header_table = {k: None for k in frequent_items.keys()}
    fptree = FPTree(header_table)
    fptree.add_transactions(sorted_transactions)
    return fptree


def conditional_pattern_base(item: str, fptree: FPTree, min_supp: int, header_table: Dict[str, int]) -> DefaultDict[Tuple[str], int]:
    """Generates the conditional base pattern for given fp tree and item. Further this method removes
    any non-frequent item from the paths (this is not a property of conditional pattern bases).
    Thereby it also populates a dictionary with frequent items and their support count.

    Args:
        item (str): The item to build a conditional base pattern on.
        fptree (FPTree): fp tree 
        min_supp (int): Minimum support as count.
        header_table (Dict[str,int]): Empty header_table, that will be populated with frequent items 
        and their count, respectively.

    Returns:
        DefaultDict[Tuple[str], int]: Count adjusted prefix-paths without item as leaf are stored as keys. 
        The count of the prefix path is stored as value.
    """
    first_item = fptree.header_table.get(item)

    paths = {}
    frequent_items = defaultdict(int)
    while first_item != None:
        leaf_with_item_label = first_item
        first_item = first_item.parent

        # Create dictionary from one path and store the path string as tuple
        path_str = tuple()
        while first_item != fptree.root:
            path_str = (first_item.item,) + path_str
            frequent_items[first_item.item] += leaf_with_item_label.count
            first_item = first_item.parent
        paths[path_str] = leaf_with_item_label.count

        first_item = leaf_with_item_label.node_link

    # Calculate frequent items over all paths
    frequent_items = {path: supp for path,
                      supp in frequent_items.items() if supp >= min_supp}
    paths_with_frequent_items = defaultdict(int)
    for items, supp in paths.items():
        adjusted_path = tuple()
        for item in items:
            if item in frequent_items:
                adjusted_path += (item,)
        if len(adjusted_path) > 0:
            paths_with_frequent_items[adjusted_path] += supp

    header_table.update(frequent_items)
    return paths_with_frequent_items


def conditional_fp_tree(pattern_base: DefaultDict[Tuple[str], int], header_table: Dict[str, int]) -> FPTree:
    """Constructs a conditional fp tree under the pattern_base for an item.
    It uses the frequent_items and their support to construct an ordered header table
    for the fp tree.

    Args:
        pattern_base (DefaultDict[Tuple[str], int]): Result of the conditional pattern base function.
        header_table (Dict[str, int]): Frequent items and their counts.

    Returns:
        FPTree: Conditional fp tree under the item, whose pattern base was provided.
    """
    header_table = {k[0]: None for k, v in sorted(
        header_table.items(), key=lambda item: item[1], reverse=True)}

    tree = FPTree(header_table)
    for path, count in pattern_base.items():
        tree.add_transaction(list(path), count)

    return tree


def generate_patterns_single_path(suffix: Tuple[str], path: Tuple[str], count: int) -> Dict[Tuple[str], int]:
    """Single path optimisation for a conditional fp tree. Builds all path combinations over the prefix path
    and appends the suffix to those. The support count is the same for all combinations as the path's count
    has been adjusted.

    Args:
        suffix (Tuple[str]): Current frequent itemset suffix.
        path (Tuple[str]): Single path in the conditional tree under the suffix.
        count (int): Support count of all path combinations.

    Returns:
        Dict[Tuple[str], int]: All path combinations concatenated to the suffix and their count.
    """
    frequent_itemsets = {}
    seeds = []

    for path_prefix in path:
        for seed in range(len(seeds)):
            frequent_itemset = seeds[seed] + (path_prefix,)
            seeds.append(frequent_itemset)
            frequent_itemsets[frequent_itemset + suffix] = count
        seeds.append((path_prefix,))
        frequent_itemsets[(path_prefix,) + suffix] = count

    return frequent_itemsets


def fp_tree_growth(fptree: FPTree, min_supp: int) -> Dict[Tuple[str], int]:
    """FP_growth algorithm to calculate all frequent itemsets satisfying the
    minimum support constraint from a given fp tree.

    Args:
        fptree (FPTree): fp tree to obtain frequent itemsets from.
        min_supp (int): Minimum support threshold as count.

    Returns:
        Dict[Tuple[str], int]: All frequent itemsets
    """
    frequent_items = {}

    def __fp_tree_growth(item_suffix: Tuple[str], fptree: FPTree, frequent_items: Dict[Tuple[str], int]):
        item = item_suffix[0]
        header_table = {}
        count = fptree.get_sum_item_counts(item)
        if count >= min_supp:
            frequent_items[item_suffix] = count
        pattern_base = conditional_pattern_base(
            item, fptree, min_supp, header_table)
        # empty set case
        if len(pattern_base) == 0:
            return
        # single path case
        if len(pattern_base) == 1:
            path, count = next(iter(pattern_base.items()))
            frequent_items.update(generate_patterns_single_path(
                item_suffix, path, count))
            return

        conditional_tree = conditional_fp_tree(pattern_base, header_table)
        for item in reversed(conditional_tree.header_table.keys()):
            __fp_tree_growth((item,) + item_suffix,
                             conditional_tree, frequent_items)

    for item in reversed(fptree.header_table.keys()):
        __fp_tree_growth((item,), fptree, frequent_items)

    return frequent_items
