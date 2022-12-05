from math import ceil, floor
from typing import Any, Dict, Iterator, Tuple
import pandas as pd
from pandas import DataFrame


def partition_intervals(num_intervals: int, attribute: str, db: DataFrame) -> pd.Series:
    """Discretizes a numerical attribute into num_intervals of equal size.

    Args:
        num_intervals (int): Number of intervals for this attribute
        attribute (str): Name of the attribute
        db (DataFrame): Database

    Returns:
        pd.Series : Series where every ajacent intervals are encoded as consecutive integers.
        The order of the intervals is reflected in the integers.
    """
    return pd.cut(
        x=db[attribute],
        bins=num_intervals,
        labels=[i for i in range(num_intervals)],
        include_lowest=True,
        retbins=True,
    )


def partition_categorical(attribute: str, db: DataFrame) -> Dict[int, Any]:
    """Maps the given categorical attribute to consecutive integers. Can also be used for
    numerical attributes.

    Args:
        attribute (str): Name of the attribute
        db (DataFrame): Database

    Returns:
        Dict[int, Any]: Mapping from category encoded as int to its categorical value
    """
    mapping = dict(zip(db[attribute].astype("category").cat.codes, db[attribute]))
    return mapping


def discretize_values(
    db: DataFrame, discretization: Dict[str, int]
) -> Tuple[Dict[str, Dict[int, Any]], DataFrame]:
    """Maps the numerical and quantititative attributes to integers as described in 'Mining Quantitative Association
    Rules in Large Relational Tables'.

    Args:
        db (DataFrame): Original Database
        discretization (Dict[str, int]): Name of the attribute (pandas column name) and the number of intervals
        for numerical attributes or 0 for categorical attributes and numerical attributes (no intervals)

    Returns:
        Tuple[Dict[str,Dict[int, Any]], DataFrame]: Encoded database and the mapping from the consecutive integers back to
        the interval / value for each attribute.
    """
    attribute_mappings = {}
    for attribute, ival in discretization.items():
        if ival == 0:
            attribute_mappings[attribute] = partition_categorical(attribute, db)
            db[attribute].replace(
                to_replace=dict(
                    zip(db[attribute], db[attribute].astype("category").cat.codes)
                ),
                inplace=True,
            )
        else:
            x, y = partition_intervals(ival, attribute, db)
            attribute_mappings[attribute] = {
                i: (ceil(y[i]), floor(y[i + 1])) for i in range(len(y) - 1)
            }
            db[attribute] = x.astype("int")

    return attribute_mappings, db


class Item:
    """Represents an item, where upper and lower are the same in case of a categorical attribute
    and lower <= upper in case of a numerical attribute with interval values.
    """

    def __init__(self, name: str, lower: int, upper: int) -> None:
        self.name = name
        self.lower = lower
        self.upper = upper

    def __lt__(self, __o: object) -> bool:
        return self.name < __o.name

    def __eq__(self, __o: object) -> bool:
        return (
            self.name == __o.name
            and self.lower == __o.lower
            and self.upper == __o.upper
        )

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return f"<{self.name}, {self.lower}, {self.upper}>"


def count_support(
    db: DataFrame, items: Dict[Tuple[Item], int], minsupp: float, drop: bool = True
) -> Dict[Tuple[Item], int]:
    """Counts the support for the given itemsets.

    Args:
        db (DataFrame): Encoded Database
        items (Dict[Tuple[Item], int]): Candidate itemsets with support count 0
        minsupp (float): minimum support threshold
        drop (bool, optional): Deletes items not having minimal support when set to true. Defaults to True.

    Returns:
        Dict[Tuple[Item], int]: Itemsets with their support
    """
    for idx, row in db.iterrows():
        for its in items.keys():
            count = 0
            for it in its:
                if row[it.name] >= it.lower and row[it.name] <= it.upper:
                    count += 1
            if count == len(its):
                items[its] += 1

    if drop:
        return {item: supp for item, supp in items.items() if supp / len(db) >= minsupp}
    else:
        return items


def find_frequent_items(
    mappings: Dict[str, Dict[int, Any]],
    db: DataFrame,
    discretizations: Dict[str, int],
    min_supp: float,
    max_supp: float,
) -> Dict[Tuple[Item], int]:
    """Generates all frequent items given the encoded database and the mappings.

    Args:
        mappings (Dict[str, Dict[int, Any]]): Attributes to their integer mapping
        db (DataFrame): Encoded Database
        discretizations (Dict[str, int]): Name of attributes to Number intervals
        min_supp (float): Minimum support for frequent itemsets
        max_supp (float): Maximum support for limiting interval merging

    Returns:
        Dict[Tuple[Item], int]: All frequent items
    """

    def merge_intervals(
        itemsets: Dict[Tuple[Item], int], max_upper: int, min_lower: int
    ) -> Dict[Tuple[Item], int]:
        """Obnoxious function to merge adjacent intervals.

        Args:
            itemsets (Dict[Tuple[Item], int]): Quantitative Attributes and their support
            max_upper (int): Max integer of interval to integer mapping
            min_lower (int): Min integer of interval to integer mapping

        Returns:
            Dict[Tuple[Item], int]: All items representing intervals, that satisfy min support
        """
        intervals = {}
        seeds = {}

        for item, supp in itemsets.items():
            norm_supp = supp / len(db)
            if norm_supp >= min_supp:
                intervals[item] = supp
            if norm_supp < max_supp:
                seeds[item] = supp

        while len(seeds) != 0:

            candidates = {}
            for item, supp in seeds.items():
                norm = supp / len(db)
                if norm >= min_supp:
                    intervals[item] = supp
                if norm < max_supp:
                    lower = item[0].lower
                    upper = item[0].upper
                    if lower > min_lower:
                        it = Item(item[0].name, lower - 1, upper)
                        for item, sup in itemsets.items():
                            if item[0].upper == lower - 1:
                                val = supp + sup
                                if candidates.get((it,)) == None:
                                    candidates[(it,)] = val
                                else:
                                    candidates[(it,)] = max(candidates[(it,)], val)
                    if upper < max_upper:
                        it = Item(item[0].name, lower, upper + 1)
                        for item, sup in itemsets.items():
                            if item[0].lower == upper + 1:
                                val = supp + sup
                                if candidates.get((it,)) == None:
                                    candidates[(it,)] = val
                                else:
                                    candidates[(it,)] = max(candidates[(it,)], val)

            seeds = candidates

        return intervals

    frequent_items = {}

    for attribute, num_intervals in discretizations.items():
        # Categorical / numerical attribute -> no intervals
        itemsets = {
            (Item(attribute, val, val),): 0 for val in mappings[attribute].keys()
        }
        itemsets = count_support(db, itemsets, min_supp, num_intervals == 0)
        if num_intervals != 0:
            itemsets = merge_intervals(
                itemsets,
                max(mappings[attribute].keys()),
                min(mappings[attribute].keys()),
            )
        frequent_items.update(itemsets)

    return frequent_items


def _generate_itemsets_by_join(
    old_itemsets: Dict[Tuple[Item], int], k: int
) -> Iterator[Tuple[str]]:
    """Joins frequent k-1 itemsets to generate k itemsets.
    It assumes the frequent k-1 itemsets are lexicographically ordered .

    Args:
        old_itemsets (Dict[Tule[Item], int]): Frequent k-1 itemsets
        k (int): The number of items that must match to join two frequent k-1 itemsets

    Yields:
        Iterator[Tuple[str]]: A candidate k itemset
    """
    for itemset in old_itemsets.keys():
        for other in old_itemsets.keys():
            skip = False
            for l in range(k - 1):
                if itemset[l] != other[l]:
                    skip = True
                    break

            # If the last attribute matches this will evaluate to false
            if not skip and itemset[k - 1] < other[k - 1]:
                yield itemset + (
                    Item(other[k - 1].name, other[k - 1].lower, other[k - 1].upper),
                )


def _is_candidate(
    old_itemsets: Dict[Tuple[Item], int], candidate_set: Tuple[Item]
) -> bool:
    """Checks whether there's any subset contained in the candidate_set, that isn't
    contained within the old_itemsets. If that is the case the candidate set can not
    be a frequent itemset and False is returned.

    Args:
        old_itemsets (List[Tuple[Item], int]): Frequent itemsets of length k
        candidate_set (Tuple[Item]): Candidate itemset with length k+1

    Returns:
        bool: True if all k-1 element subsets of candidate_set are contained within old_itemsets.
    """
    if len(candidate_set) == 2:
        return True

    for i in range(len(candidate_set)):
        if not candidate_set[0:i] + candidate_set[i + 1 :] in old_itemsets:
            return False

    return True


def quantitative_itemsets(
    db: DataFrame,
    discretization: Dict[str, int],
    minsupp: float = 0.05,
    maxsupp: float = 0.1,
):
    mappings, encoded_db = discretize_values(db.copy(deep=True), discretization)
    frequent_items = find_frequent_items(
        mappings, encoded_db, discretization, minsupp, maxsupp
    )
    frequent_k_itemsets = frequent_items.copy()
    k = 1

    while len(frequent_k_itemsets) != 0:
        candidates = {}
        print(k)
        for candidate_set in _generate_itemsets_by_join(frequent_k_itemsets, k):
            if _is_candidate(frequent_k_itemsets, candidate_set):
                candidates[candidate_set] = 0

        frequent_k_itemsets = count_support(encoded_db, candidates, minsupp)

        frequent_items.update(frequent_k_itemsets)
        k += 1

    return frequent_items
