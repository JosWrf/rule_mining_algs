from typing import Dict, List
import numpy as np

from pandas import DataFrame

from algs.util import get_frequent_1_itemsets


class FPNode:
    def __init__(self, item: str, parent: "FPNode", count: int = 1) -> None:
        self.node_link = None
        self.parent = parent
        self.item = item
        self.count = count
        self.children = {}

class FPTree:
    def __init__(self, header_table: Dict[str, None]) -> None:
        self.root = FPNode(None, None)
        self.header_table = header_table 

    def add_transaction(self, transaction: List[str]) -> None:

        def __add_transaction(depth: 0, node: "FPNode") -> None:
            if depth == len(transaction): 
                return 
            
            item_name = transaction[depth]
            child = node.children.get(item_name)
            if child != None:
                child.count += 1

            else:
                child = FPNode(item_name, node)
                node.children[item_name] = child
                self.__set_node_link(item_name, child)      

            __add_transaction(depth + 1, child)

        __add_transaction(0, self.root)

    def __set_node_link(self, item_name: str, node: "FPNode") -> None:
        next_node = self.header_table.get(item_name)
        if next_node == None:
            self.header_table[item_name] = node
        
        else:
            while next_node != None:
                previous_node = next_node
                next_node = next_node.node_link
            
            previous_node.node_link = node


    def add_transactions(self, transactions: DataFrame) -> None:
        for row in range(len(transactions)):
            transaction = list(transactions.loc[row, list(transactions.loc[row])].index)
            self.add_transaction(transaction)


def fp_tree(transactions: DataFrame, min_support: float = 0.05) -> DataFrame:
    # Get frequent items and sort transactions
    items = np.array(transactions.columns)
    frequent_items = get_frequent_1_itemsets(items, transactions, min_support)
    frequent_items = {k[0]: v for k, v in sorted(frequent_items.items(), key=lambda item: item[1], reverse=True)}
    sorted_transactions = get_transformed_dataframe(transactions, items, frequent_items)

    # Build header table for node links and construct FP tree
    header_table = {k : None for k in frequent_items.keys()}
    fptree = FPTree(header_table)
    fptree.add_transactions(sorted_transactions)


def get_transformed_dataframe(old_df: DataFrame, all_items: np.ndarray, frequent_items: Dict[str, float]) -> DataFrame:
    drop_columns = [item for item in all_items if not frequent_items.get(item)]
    return old_df.drop(drop_columns, inplace=False, axis=1)[frequent_items.keys()]

def fp_growth():
    pass