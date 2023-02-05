from math import ceil, floor
from typing import Any, Dict, Iterator, Set, Tuple
import numpy as np

import pandas as pd
from mlxtend.preprocessing import TransactionEncoder
from pandas import DataFrame
from sklearn.cluster import Birch


def partition_intervals(
    num_intervals: int, attribute: str, db: DataFrame, equi_depth: bool
) -> pd.Series:
    """Discretizes a numerical attribute into num_intervals of equal size/ width.

    Args:
        num_intervals (int): Number of intervals for this attribute
        attribute (str): Name of the attribute
        db (DataFrame): Database
        equi_depth (bool): Equi-depth discretization else equi-width

    Returns:
        pd.Series : Series where every ajacent intervals are encoded as consecutive integers.
        The order of the intervals is reflected in the integers.
    """
    if equi_depth:
        # Determine the new number of labels
        _, y = pd.qcut(
            x=db[attribute],
            q=num_intervals,
            retbins=True,
            duplicates="drop")
        return pd.qcut(
            x=db[attribute],
            q=num_intervals,
            labels=[i for i in range(len(y)-1)],
            retbins=True,
            duplicates="drop"
        )
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
    mapping = dict(zip(db[attribute].astype(
        "category").cat.codes, db[attribute]))
    return mapping


def discretize_values(
    db: DataFrame, discretization: Dict[str, int], equi_depth: bool,
) -> Tuple[Dict[str, Dict[int, Any]], DataFrame]:
    """Maps the numerical and quantititative attributes to integers as described in 'Mining Quantitative Association
    Rules in Large Relational Tables'.

s:
        db (DataFrame): Original Database
        discretization (Dict[str, int]): Name of the attribute (pandas column name) and the number of intervals
        for numerical attributes or 0 for categorical attributes and numerical attributes (no intervals)
        equi_depth (bool): Equi-depth discretization else equi-width.

    Returns:
        Tuple[Dict[str,Dict[int, Any]], DataFrame]: Encoded database and the mapping from the consecutive integers back to
        the interval / value for each attribute.
    """
    attribute_mappings = {}
    for attribute, ival in discretization.items():
        if ival == 0:
            attribute_mappings[attribute] = partition_categorical(
                attribute, db)
            db[attribute].replace(
                to_replace=dict(
                    zip(db[attribute], db[attribute].astype(
                        "category").cat.codes)
                ),
                inplace=True,
            )
        else:
            x, y = partition_intervals(ival, attribute, db, equi_depth)
            int_val = pd.api.types.is_integer_dtype(db[attribute])
            attribute_mappings[attribute] = {
                i: (
                    ceil(y[i]) if int_val else y[i],
                    floor(y[i + 1]) if int_val else y[i + 1],
                )
                for i in range(len(y) - 1)
            }
            db[attribute] = x.astype("int")

    return attribute_mappings, db


def static_discretization(db: DataFrame, discretization: Dict[str, int], equi_depth: bool = False) -> DataFrame:
    """Discretizes all attributes in the dataframe. It thereby reduces the problem of mining
    quantitative itemsets to the problem of mining itemsets over binary data.

    Args:
        db (DataFrame): Dataframe to be transformed
        discretization (Dict[str, int]): Name of the attribute (pandas column name) and the number of intervals
        equi_depth (bool): Equi-depth discretization else equi-width (Defaults to False).

    Returns:
        DataFrame: DataFrame, where all columns correspond to binary attributes
    """
    mappings, encoded_db = discretize_values(
        db.copy(deep=True), discretization, equi_depth)
    return _static_discretization(encoded_db, mappings)


def _static_discretization(
    encoded_db: DataFrame, mapped_vals: Dict[str, Dict[int, Any]]
) -> DataFrame:
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
                name = f"{attribute} = {val[0]}..{val[1]}"
            else:
                name = f"{attribute} = {val}"

            row_entry.append(name)

        rows.append(row_entry)

    te = TransactionEncoder()
    te_ary = te.fit_transform(rows)
    df = pd.DataFrame(te_ary, columns=te.columns_)
    return df


def cluster_interval_data(db: DataFrame, attr_threshold: Dict[Tuple[str], float], num_clusters: bool = False) -> DataFrame:
    """Clusters interval data, using the birch clustering algorithm as described in
    'Association Rules over Interval Data'. The threshold is the upper bound of the
    radius of subclusters. Further the clusters are described by their smallest bounding box.

    Args:
        db (DataFrame): Dataset, to mine quantitative association rules from
        attr_threshold (Dict[Tuple[str], float]): Maps attribute (sets) to their radius threshold, which
        in turn determines the cluster quality. If num_clusters is set, the determined number of 
        clusters is generated instead.
        num_clusters (bool): If set to True the thresholds are interpreted as number of clusters. 

    Returns:
        DataFrame: One column for each attribute, value pair of all attributes (after clustering).
        In the case of clusters the values are bounding boxes to represent the cluster.
    """
    data = db.copy(deep=True)

    names = [name for tpl in attr_threshold.keys() for name in tpl]
    discretization = {name: 0 for name in data.columns if name not in names}
    for attributes, radius in attr_threshold.items():
        # Build name of the attribute (can be combined e.g. x,y)
        name = "{" + attributes.__str__()[1:-1].replace("'", "") + "}"
        name = name.replace(",", "") if len(attributes) == 1 else name
        discretization[name] = 0
        attributes = list(attributes)

        # Use birch clustering alg and calculate bounding boxes to represent clusters
        brc = Birch(n_clusters=int(radius) if num_clusters else None,
                    threshold=0.5 if num_clusters else radius, copy=True)
        data[name] = brc.fit_predict(data[attributes])
        mins = data.groupby(name).min(numeric_only=True)[
            attributes].to_dict("tight")
        maxs = data.groupby(name).max(numeric_only=True)[
            attributes].to_dict("tight")

        # Map the cluster id to a name representing the cluster
        replace_dict = {}
        idx = 0
        for d1, d2 in zip(mins["data"], maxs["data"]):
            attr_name = d1.__str__() + " x " + d2.__str__()
            replace_dict[idx] = attr_name
            idx += 1

        data[name].replace(replace_dict, inplace=True)
        data.drop(labels=attributes, axis=1, inplace=True)
    return static_discretization(data, discretization)


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
            __o.lower > self.lower and __o.upper < self.upper or self.lower >= __o.upper
        ):  # Inclusion relation would cause a split in 2 non-adjecent subintervals
            return None
        if (
            __o.lower == self.lower and __o.upper == self.upper
        ):  # Same interval -> categorical
            return __o
        if self.lower == __o.lower:  # [5,8] - [5,6] = [6,8]
            return Item(self.name, __o.upper, self.upper)
        else:  # [5,8] - [7,8] = [5,7]
            return Item(self.name, self.lower, __o.lower)

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
    for its in items.keys():
        conditions = [(db[it.name] >= it.lower) & (
            db[it.name] <= it.upper) for it in its]
        mask = np.column_stack(conditions).all(axis=1)
        items[its] = mask.sum()

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

        while seeds:

            candidates = {}
            for item, supp in seeds.items():
                norm = supp / len(db)
                if norm >= min_supp:
                    intervals[item] = supp
                if norm < max_supp:
                    lower, upper = item[0].lower, item[0].upper
                    if lower > min_lower:
                        it = Item(item[0].name, lower - 1, upper)
                        for item, sup in itemsets.items():
                            if item[0].upper == lower - 1:
                                val = supp + sup
                                if candidates.get((it,)) == None:
                                    candidates[(it,)] = val
                                else:
                                    candidates[(it,)] = max(
                                        candidates[(it,)], val)
                    if upper < max_upper:
                        it = Item(item[0].name, lower, upper + 1)
                        for item, sup in itemsets.items():
                            if item[0].lower == upper + 1:
                                val = supp + sup
                                if candidates.get((it,)) == None:
                                    candidates[(it,)] = val
                                else:
                                    candidates[(it,)] = max(
                                        candidates[(it,)], val)

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
            # Attributes(X) != Attributes(X')
            if items[i].name != itemset[i].name:
                attrs = False
                break
            if (
                items[i] == itemset[i]
            ):  # Having the same boundaries, implies a categorical attribute
                continue
            elif items[i].lower == items[i].upper and itemset[i].lower == itemset[i].upper and items[i].lower != itemset[i].lower:
                attrs = False
                break
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
        if (new_items[specialization] / n / exp_supp) < R:
            return False

    return True


def remove_r_uninteresting_itemsets(
    db: DataFrame, frequent_itemsets: Dict[Tuple[Item], int], R: float
) -> Tuple[Dict[Tuple[Item], int], Dict[Tuple[Item], int]]:
    """Uses the definition of R-interestingness of itemsets in the context of
    quantitative association rules to prune itemsets, that do not fullfill it.

    Args:
        db (DataFrame): Transformed Database
        frequent_itemsets (Dict[Tuple[Item], int]): All mined frequent itemsets
        R (float): Interest Level

    Returns:
        Tuple[Dict[Tuple[Item], int], Dict[Tuple[Item], int]]: Position[0]: Frequent and R-interesting itemsets.
        Position[1]: Itemsets that are not R-interesting.
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
    elements_to_remove = {}

    for item, support in frequent_itemsets.items():
        partial_order = get_generalizations_specializations(
            frequent_itemsets, item)

        interesting = True
        sub_intervals, sub_items = _get_subintervals(
            db, partial_order[0], item)
        for gen, supp in partial_order[1].items():
            if not _is_r_interesting(gen, item) or not _is_specialization_interesting(
                sub_intervals, gen, sub_items, frequent_itemsets, R, supp / n, n
            ):
                interesting = False
                elements_to_remove[item] = support
                break

        if interesting:
            r_interesting_itemsets[item] = support

    return r_interesting_itemsets, elements_to_remove


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
    new_candidates = {}

    for itemset in old_itemsets.keys():
        for other in old_itemsets.keys():
            if all(itemset[i] == other[i] for i in range(k-1)) and itemset[k-1] < other[k-1]:
                new_candidates[
                    itemset + (
                        Item(other[k - 1].name, other[k - 1].lower,
                             other[k - 1].upper),
                    )] = 0
    return new_candidates


def _downward_closure(old_itemsets: Dict[Tuple[Item], int], candidates: Dict[Tuple[Item], int]) -> Dict[Tuple[Item], int]:
    """Uses the downward closure property of support to prune any k-itemsets, which do 
    have at least one k-1 itemset, which is not frequent.

    Args:
        old_itemsets (Dict[Tuple[Item], int]): Frequenkt k-1 itemsets
        candidates (Dict[Tuple[Item], int]): Potential k itemsets

    Returns:
        Dict[Tuple[Item], int]: Pruned potential k itemsets
    """
    result = {}
    for candidate in candidates:
        found = all(candidate[0:i] + candidate[i + 1:]
                    in old_itemsets for i in range(len(candidate)))
        if found:
            result[candidate] = 0
    return result


def quantitative_itemsets(
    db: DataFrame,
    discretization: Dict[str, int],
    minsupp: float = 0.05,
    maxsupp: float = 0.1,
    R: float = 0.0,
    equi_depth: bool = False,
) -> DataFrame:
    """ Provides an algorithm similar to the one introduced in 
    'Mining Quantitative Association Rules in Large Relational Tables'.
    Optimizations for support counting are omitted, however.

    Args:
        db (DataFrame): Data mining context
        discretization (Dict[str, int]): Attributes and how they should be discretized. 0 indicates no
        merging of intervals. Any number greater than 0 will yield the amount of intervals for this attribute, for 
        the initial partitioning.
        minsupp (float, optional): Min support threshold. Defaults to 0.05.
        maxsupp (float, optional): Max support threshold for interval merging. Defaults to 0.1.
        R (float, optional): R Interest Level. Defaults to 0.0. If left at 0.0 no R-interestingess pruning 
        occurs.
        equi_depth (bool, optional): Equi-depth intervals when True else equi-width intervals. Defaults to False.

    Returns:
        DataFrame: All quantitative itemsets satisfying the given constraints.
    """
    mappings, encoded_db = discretize_values(
        db.copy(deep=True), discretization, equi_depth)
    frequent_items = find_frequent_items(
        mappings, encoded_db, discretization, minsupp, maxsupp
    )
    frequent_items = _prune_by_r_interest(
        frequent_items, discretization, R, len(db))
    frequent_k_itemsets = frequent_items.copy()
    k = 1
    to_remove = {}

    while len(frequent_k_itemsets) != 0:
        candidates = _generate_itemsets_by_join(frequent_k_itemsets, k)
        candidates = _downward_closure(frequent_k_itemsets, candidates)

        frequent_k_itemsets = count_support(encoded_db, candidates, minsupp)

        frequent_items.update(frequent_k_itemsets)
        k += 1

    if R != 0:
        frequent_items, to_remove = remove_r_uninteresting_itemsets(
            encoded_db, frequent_items, R)

    itemsets = []
    for itemset, support in {**frequent_items, **to_remove}.items():
        items = []
        for item in itemset:
            vals = mappings[item.name]
            lower = vals[item.lower]
            upper = vals[item.upper]
            item_value = f"{lower[0]}..{upper[1]}" if discretization[item.name] else f"{lower}"
            items.append(f"{item.name} = {item_value}")
        itemsets.append({"itemsets": tuple(
            items), "support": support / len(db), "ignore": itemset in to_remove})

    return pd.DataFrame(itemsets)
