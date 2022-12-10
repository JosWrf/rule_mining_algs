from math import ceil, floor
from typing import Any, Dict, Iterator, Set, Tuple
import pandas as pd
from pandas import DataFrame
from mlxtend.preprocessing import TransactionEncoder


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
            int_val = db[attribute].dtype == int
            attribute_mappings[attribute] = {
                i: (
                    ceil(y[i]) if int_val else y[i],
                    floor(y[i + 1]) if int_val else y[i + 1],
                )
                for i in range(len(y) - 1)
            }
            db[attribute] = x.astype("int")

    return attribute_mappings, db

def static_discretization(db: DataFrame, discretization: Dict[str, int]) -> DataFrame:
    """Discretizes all attributes in the dataframe. It thereby reduces the problem of mining
    quantitative itemsets to the problem of mining itemsets over binary data.

    Args:
        db (DataFrame): Dataframe to be transformed
        discretization (Dict[str, int]): Name of the attribute (pandas column name) and the number of intervals

    Returns:
        DataFrame: DataFrame, where all columns correspond to binary attributes
    """
    mappings, encoded_db = discretize_values(db.copy(deep=True), discretization)
    return _static_discretization(encoded_db, mappings)

def _static_discretization(encoded_db: DataFrame, mapped_vals: Dict[str, Dict[int, Any]]) -> DataFrame:
    """Discretizes all attributes in the dataframe.

    Args:
        encoded_db (DataFrame): Transformed database, where each value / interval is represented by an integer
        mapped_vals (Dict[str, Dict[int, Any]]): Stores the information of the value transformations for each attribute

    Returns:
        DataFrame: DataFrame, where all columns correspond to binary attributes
    """
    rows = []
    for idx, row in encoded_db.iterrows():
        row_entry = []
        attributes = row.index.array
        for attribute in attributes:
            name = ""
            val = mapped_vals[attribute][row[attribute]]
            if type(val) == tuple:
                name = f"{attribute} = <{val[0]}..{val[1]}>"
            else:
                name = f"{attribute} = {val}"
            
            row_entry.append(name)

        rows.append(row_entry)
    
    te = TransactionEncoder()
    te_ary = te.fit_transform(rows)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    return df
    


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

    def is_generalization(self, other: object) -> bool:
        return other.lower >= self.lower and other.upper <= self.upper

    def is_specialization(self, other: object) -> bool:
        return other.lower <= self.lower and other.upper >= self.upper

    def __sub__(self, __o: object) -> object:
        if (
            __o.lower > self.lower and __o.upper < self.upper
        ):  # Inclusion relation would cause a split in 2 non-adjecent subintervals
            return None
        if (
            __o.lower == self.lower and __o.upper == self.upper
        ):  # Same interval -> categorical
            return __o
        if self.lower == __o.lower:  # [5,8] - [5,6] = [7,8]
            return Item(self.name, __o.lower + 1, self.upper)
        else:  # [5,8] - [7,8] = [5,6]
            return Item(self.name, self.lower, __o.upper - 1)

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


def _prune_by_r_interest(
    frequent_items: Dict[Tuple[Item], int],
    discretizations: Dict[str, int],
    R: float,
    n: int,
) -> Dict[Tuple[Item], int]:
    """Prunes all quantitative attributes with support/n > 1/R (Lemma 5)

    Args:
        frequent_items (Dict[Tuple[Item], int]): Frequent items
        discretizations (Dict[str, int]): Name of Attributes to num intervals
        R (float): R-Interest
        n (int): Number of entries in the db

    Returns:
        Dict[Tuple[Item], int]: All items whose fractional support does not exceed 1/R
    """
    if R == 0:
        return frequent_items
    return {
        item: supp
        for item, supp in frequent_items.items()
        if discretizations[item[0].name] == 0 or supp / n <= 1 / R
    }


def get_generalizations_specializations(
    frequent_itemsets: Dict[Tuple[Item], int], itemset: Tuple[Item]
) -> Dict[int, Dict[Tuple[Item], int]]:
    """Determines all generalizations and specializations of the given itemset.

    Args:
        frequent_itemsets (Dict[Tuple[Item], int]): All frequent itemsets.
        itemset (Tuple[Item]): Itemset, whose generalizations and specializations are to be determined.

    Returns:
        Dict[int, Dict[Tuple[Item], int]]: The key 0 maps to all specializations of the itemset and the key 1
        gives all generalizations of the itemset.
    """
    result = {0: {}, 1: {}}
    for items, supp in frequent_itemsets.items():
        if len(items) != len(itemset):  # Attributes(X) != Attributes(X')
            continue
        found_spec = 0
        found_gen = 0
        attrs = True

        for i in range(len(items)):
            if items[i].name != itemset[i].name:  # Attributes(X) != Attributes(X')
                attrs = False
                break
            if (
                items[i] == itemset[i]
            ):  # Having the same boundaries, implies a categorical attribute
                continue
            elif itemset[i].is_generalization(items[i]):
                found_spec = 1
            elif itemset[i].is_specialization(items[i]):
                found_gen = 1

        if found_gen + found_spec != 1 or not attrs:
            continue
        elif found_spec:
            result[0][items] = supp
        else:
            result[1][items] = supp

    return result


def _get_subintervals(
    db: DataFrame, specializations: Dict[Tuple[Item], int], itemset: Tuple[int]
) -> Tuple[Set[Tuple[Item]], Dict[Tuple[Item], int]]:
    """Calculates the difference of an itemset to all its specializations.

    Args:
        db (DataFrame): Transformed Database
        specializations (Dict[Tuple[Item], int]): All specializations of the given itemset
        itemset (Tuple[int]): Itemset to substract a specialization from

    Returns:
        Tuple[Set[Tuple[Item]], Dict[Tuple[Item], int]]: Itemsets generated from the difference,
        all individual items that were generated from the difference and their support aswell as 
        the itemsets themselves. 
    """
    new_itemsets = set()  # Holds X-X'
    new_items = {}  # Holds the items that are created by X-X'

    for items in specializations.keys():
        new_itemset = []
        for i in range(len(items)):
            sub_interval = itemset[i] - items[i]
            if sub_interval is None:
                break
            else:
                new_items.update(
                    {(sub_interval,): 0}
                )  # We need the support for individual elements
                new_itemset.append(sub_interval)

        if len(new_itemset) == len(itemset):
            new_itemsets.add(tuple(new_itemset))
            new_items.update(
                {tuple(new_itemset): 0}
            )  # We need the support for all X-X' aswell

    new_items = count_support(db, new_items, 0.0, False)
    return new_itemsets, new_items


def _is_specialization_interesting(
    specializations: Set[Tuple[Item]],
    generalization: Tuple[Item],
    new_items: Dict[Tuple[Item], int],
    frequent_itemsets: Dict[Tuple[Item], int],
    R: float,
    gen_supp: float,
    n: int,
) -> bool:
    """Determine whether the difference (X-X') from the itemset to any of its specializations
    is r-interesting wrt. the generalization of the itemset. 

    Args:
        specializations (Set[Tuple[Item]]): All itemsets of the form: X-X'
        generalization (Tuple[Item]): The generalization of the itemset
        new_items (Dict[Tuple[Item], int]): Items/Itemsets from (X-X') with support information
        frequent_itemsets (Dict[Tuple[Item], int]): All mined frequent itemsets
        R (float): Interest level
        gen_supp (float): Support for the generalization
        n (int): Number of transactions in the database

    Returns:
        bool: False if there's any specialization of X' st. X-X' is not r-interesting.
    """
    if len(specializations) == 0:
        return True

    for specialization in specializations:
        exp_supp = gen_supp
        for i in range(len(specialization)):
            exp_supp *= (
                new_items[(specialization[i],)]
                / frequent_itemsets[(generalization[i],)]
            )
        if new_items[specialization] / n / exp_supp < R:
            return False

    return True


def remove_r_uninteresting_itemsets(
    db: DataFrame, frequent_itemsets: Dict[Tuple[Item], int], R: float
) -> Dict[Tuple[Item], int]:
    """Uses the definition of R-interestingness of itemsets in the context of 
    quantitative association rules to prune itemsets, that do not fullfill it.

    Args:
        db (DataFrame): Transformed Database
        frequent_itemsets (Dict[Tuple[Item], int]): All mined frequent itemsets
        R (float): Interest Level

    Returns:
        Dict[Tuple[Item], int]: Frequent and R-interesting itemsets. 
    """
    def _is_r_interesting(generalization: Tuple[Item], itemset: Tuple[Item]) -> bool:
        """Indicates whether the support of the itemset is r times the expected support
        given its generalization.

        Args:
            generalization (Tuple[Item]): Generalization of the itemset
            itemset (Tuple[Item]): Potentially r-interesting itemset

        Returns:
            bool: True if the itemset is r-interesting wrt. to its generalization else False
        """
        n = len(db)
        exp_supp = frequent_itemsets[generalization] / n
        for i in range(len(generalization)):
            exp_supp *= (
                frequent_itemsets[(itemset[i],)]
                / frequent_itemsets[(generalization[i],)]
            )
        return (frequent_itemsets[itemset] / n / exp_supp) >= R

    n = len(db)
    r_interesting_itemsets = {}
    for item, support in frequent_itemsets.items():
        partial_order = get_generalizations_specializations(frequent_itemsets, item)

        interesting = True
        sub_intervals, sub_items = _get_subintervals(db, partial_order[0], item)
        for gen, supp in partial_order[1].items():
            if not _is_r_interesting(gen, item) or not _is_specialization_interesting(
                sub_intervals, gen, sub_items, frequent_itemsets, R, supp / n, n
            ):
                interesting = False
                break

        if interesting:
            r_interesting_itemsets[item] = support

    return r_interesting_itemsets


def quantitative_itemsets(
    db: DataFrame,
    discretization: Dict[str, int],
    minsupp: float = 0.05,
    maxsupp: float = 0.1,
    R: float = 0.0,
) -> DataFrame:
    mappings, encoded_db = discretize_values(db.copy(deep=True), discretization)
    frequent_items = find_frequent_items(
        mappings, encoded_db, discretization, minsupp, maxsupp
    )
    frequent_items = _prune_by_r_interest(frequent_items, discretization, R, len(db))
    frequent_k_itemsets = frequent_items.copy()
    k = 1

    while len(frequent_k_itemsets) != 0:
        candidates = {}
        for candidate_set in _generate_itemsets_by_join(frequent_k_itemsets, k):
            if _is_candidate(frequent_k_itemsets, candidate_set):
                candidates[candidate_set] = 0

        frequent_k_itemsets = count_support(encoded_db, candidates, minsupp)

        frequent_items.update(frequent_k_itemsets)
        k += 1

    if R != 0:
        frequent_items = remove_r_uninteresting_itemsets(encoded_db, frequent_items, R)

    # Map resulting itemsets back to their (interval) values
    itemsets = []
    for itemset, support in frequent_items.items():
        items = []
        for item in itemset:
            vals = mappings[item.name]
            lower = vals[item.lower]
            upper = vals[item.upper]
            if discretization[item.name] == 0:
                assert lower == upper
                items.append(f"{item.name} = {lower}")
            else:
                items.append(f"{item.name} = {lower[0]}..{upper[1]}")
        itemsets.append({"itemsets": tuple(items), "support": support / len(db)})

    df = pd.DataFrame(itemsets)
    return df