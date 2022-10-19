from typing import Dict, List, Tuple


class HashTree:
    def __init__(self, depth: int = 0, leaf: bool = True, max_size: int = 57) -> None:
        self.children = {}
        self.itemsets = {}
        self.leaf = leaf
        self.max_size = max_size
        self.depth = depth

    def add_itemset(self, itemset: Tuple[str]) -> None:
        """Adds the given itemset to the hash tree. It is assumed that no duplicates occur
        and no differently sized itemsets are inserted.
        When max_size is exceeded at a leaf node, that node is converted to an inner node, 
        unless its already at depth equal to the length of the itemset.

        Args:
            itemset (Tuple[str]): The itemset to add to the tree.
        """
        if (self.leaf and self.max_size > len(self.itemsets)) or self.depth == len(
            itemset
        ):
            self.itemsets[itemset] = 0

        else:
            # Hash to some value to navigate to the child
            if not self.leaf:
                hash_value = self.hash_func(itemset, self.depth)
                if self.children.get(hash_value) == None:
                    self.children[hash_value] = HashTree(
                        self.depth + 1, max_size=self.max_size
                    )
                self.children[hash_value].add_itemset(itemset)

            # Make the leaf an inner node
            else:
                self.itemsets[itemset] = 0
                for items in self.itemsets.keys():
                    hash_value = self.hash_func(items, self.depth)
                    if self.children.get(hash_value) == None:
                        self.children[hash_value] = HashTree(
                            self.depth + 1, max_size=self.max_size
                        )
                    self.children[hash_value].add_itemset(items)

                self.itemsets = {}
                self.leaf = False

    def hash_func(self, itemset: Tuple[str], depth: int) -> int:
        """Hash function used for hashing items in itemsets.

        Args:
            itemset (Tuple[str]): The itemset to hash
            depth (int): The position of the item to apply the hash function to

        Returns:
            int: Hash value of the hashed item.
        """
        return sum(ord(item) for item in itemset[depth]) % 7

    def transaction_counting(
        self,
        transaction: List[str],
        lower_boundary: int,
        k: int,
        visited: Dict["HashTree", bool],
    ) -> None:
        """Traverses the hash tree, given a transaction. The transaction is
        recursively hashed. Upon encountering a leaf the transaction is matched
        against all stored itemsets. If any of these are a subset of the transaction
        their support count is increase by one.

        Args:
            transaction (List[str]): Transaction to match against candidates
            lower_boundary (int): Index of the item in the transaction that is to be hashed
            k (int): Length of itemsets
            visited (Dict[HashTree]): Stores the already visited leaves of the tree, as to not double down
            on counting the same itemset for one transaction.
        """
        if self.leaf:
            if visited.get(self) != None:
                return

            for itemset in self.itemsets.keys():

                if set(itemset).issubset(transaction):
                    self.itemsets[itemset] += 1
            visited[self] = True

        else:
            for i in range(lower_boundary, len(transaction) - k + self.depth + 1):
                hash_value = self.hash_func(transaction, i)
                child = self.children.get(hash_value)
                if child:
                    child.transaction_counting(transaction, i + 1, k, visited)

    def get_frequent_itemsets(
        self, min_support: float, transaction_count: int
    ) -> Dict[Tuple[str], float]:
        """Finds all itemsets in the tree, whose count / |transactions| >= min_support and
        returns them.

        Args:
            min_support (float): Minimum support
            transaction_count (int): Number of transactions in the database

        Returns:
            Dict[Tuple[str], float]: Dictionary containing pairs of itemsets, satisfying the minimum
            support constraint and their support.
        """
        if self.leaf:
            return {
                itemset: support / transaction_count
                for itemset, support in self.itemsets.items()
                if support / transaction_count >= min_support
            }

        else:
            itemsets = {}
            for child_node in self.children.values():
                itemsets.update(
                    child_node.get_frequent_itemsets(min_support, transaction_count)
                )

            return itemsets

    def get_all_itemsets(self) -> Dict[Tuple[str], int]:
        """Returns all stored itemsets and their counts.
        Note: This method should only be used for testing

        Returns:
            Dict[Tuple[str], int]: Dictionary with itemset, count pairs.
        """
        if self.leaf:
            return self.itemsets

        else:
            result = {}
            for child in self.children.values():
                result.update(child.get_all_itemsets())
            return result

    def number_items(self) -> int:
        """Returns the number of itemsets stored in the tree

        Returns:
            int: Number of itemsets
        """
        if self.leaf:
            return len(self.itemsets)

        items = 0
        for child in self.children.values():
            items += child.number_items()
        return items
