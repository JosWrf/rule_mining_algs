from itertools import chain, combinations
from typing import Any, Dict, Iterator, List, Tuple
from pandas import DataFrame, Series

from algs.util import (
    confidence,
    conviction,
    cosine,
    imbalance_ratio,
    independent_cosine,
    kulczynski,
    lift,
    measure_dict,
)


def generate_rules(frequent_itemsets: DataFrame, min_conf: float = 0.5) -> DataFrame:
    """Generates all rules that satisfy the minimum confidence constraint for all frequent itemsets.
    This algorithm is described in 'Fast Algorithms for Mining Association Rules'
    on p.14.


    Args:
        frequent_itemsets (DataFrame): Frequent itemsets, which were found by e.g. using the apriori algorithm
        min_conf (float, optional): Minimum confidence threshold. Defaults to 0.5.

    Returns:
        DataFrame: All rules satisfying the constraints.
    """
    support_mapping = frequent_itemsets.set_index(
        "itemsets").to_dict()["support"]

    def __ap_genrules(
        itemset: Series, consequents: List[Tuple[str]], m: int
    ) -> Iterator[Dict[str, Any]]:
        """Checks the minimum confidence constraint for all rules that can be built with the consequents
        in the consequents argument and yields them. The consequences are extended as long as the size is smaller than
        the size of the corresponding itemset and the frontier is not empty.

        Args:
            itemset (Series): The itemset along its support
            consequents (List[Tuple[str]]): List of all candidate consequents, which may give rules that have minimum confidence
            m (int): The size of the elements contained in consequents.

        Yields:
            Iterator[Dict[str, Any]]
                ]: Rule antecedents and consequents with objective measures
        """
        new_consequents = []
        for consequent in consequents:
            support_rule = itemset["support"]
            if support_rule == 0:
                continue
            antecedent = tuple(
                [item for item in itemset["itemsets"] if item not in consequent]
            )
            conf = confidence(support_mapping[antecedent], support_rule)
            if conf >= min_conf:
                new_consequents.append(consequent)
                yield {
                    "antecedents": antecedent,
                    "consequents": consequent,
                    "support": support_rule,
                    "confidence": conf,
                    "cosine": cosine(
                        support_mapping[antecedent],
                        support_mapping[consequent],
                        support_rule,
                    ),
                    "idependent_cosine": independent_cosine(
                        support_mapping[antecedent], support_mapping[consequent]
                    ),
                    "lift": lift(
                        support_mapping[antecedent],
                        support_mapping[consequent],
                        support_rule,
                    ),
                    "conviction": conviction(
                        support_mapping[antecedent],
                        support_mapping[consequent],
                        support_rule,
                    ),
                    "imbalance_ratio": imbalance_ratio(
                        support_mapping[antecedent],
                        support_mapping[consequent],
                        support_rule,
                    ),
                    "kulczynksi": kulczynski(
                        support_mapping[antecedent],
                        support_mapping[consequent],
                        support_rule,
                    ),
                }

        if len(itemset["itemsets"]) > m + 1:
            yield from __ap_genrules(
                itemset, __apriori_gen(new_consequents, m - 1), m + 1
            )

    rules = []
    for itemsets in frequent_itemsets.iterrows():
        itemset = itemsets[1]["itemsets"]
        # Some algorithms prune itemsets, but their support information would still be
        # required. This itemsets are added to the df but ignore is True for them.
        if "ignore" in frequent_itemsets.columns and itemsets[1]["ignore"] == True:
            continue
        if len(itemset) >= 2:
            consequents = __get_1_item_consequents(itemset)
            for rule in __ap_genrules(itemsets[1], consequents, 1):
                rules.append(rule)

    df = DataFrame(
        rules,
        index=[i for i in range(len(rules))],
    )

    return df


def __get_1_item_consequents(itemsets: List[str]) -> List[Tuple[str]]:
    """Calculates the consequents for frequent itemsets consisting of 1 element.

    Args:
        itemsets (List[str]): Frequent itemset

    Returns:
        List[Tuple[str]]: List of consequents, where each tuple contains one item.
    """
    return [(itemsets[i],) for i in range(len(itemsets))]


def __apriori_gen(old_candidates: List[Tuple[str]], k: int) -> List[Tuple[str]]:
    """Similar to the apriori gen method, this algorithm merges consequents of the previous
    pass satisfying the minimum confidence constraint to generate new candidate consequences
    and thus new rules.

    Args:
        old_candidates (List[Tuple[str]]): List of k element consequences in the last pass.
        k (int): Number of elements that are supposed to match, when joining two consequents of the last pass.

    Returns:
        List[Tuple[str]]: Consequents with size of k+2, where k refers to the size of the input parameter.
    """
    candidates = set()
    for i in range(len(old_candidates)):
        for j in range(i + 1, len(old_candidates)):
            skip = False
            for l in range(k - 1):
                if old_candidates[i][l] != old_candidates[j][l]:
                    skip = True
                    break

            if not skip and old_candidates[i][k - 1] < old_candidates[j][k - 1]:
                candidates.add(old_candidates[i] + (old_candidates[j][k - 1],))

    cands = [
        candidate
        for candidate in candidates
        if all(
            candidate[:i] + candidate[i + 1:] in old_candidates
            for i in range(len(candidate))
        )
    ]
    return cands


def minimal_non_redundant_rules(
    closed_frequent_itemsets: DataFrame, min_conf: float = 0.5
) -> DataFrame:
    """Determines the set of minimal non redundant rules by first calculating the generic basis and then
    the transitive reduction of the informative basis, all according to 'Mining minimal non-redundant
    association rules'.

    Args:
        closed_frequent_itemsets (DataFrame): All frequent closed itemsets and their generators as determined
        by the AClose algorithm.
        min_conf (float, optional): Minimum confidence threshold. Defaults to 0.5.

    Returns:
        DataFrame: Minimal non-redundant association rules with confidence, support, antecedents and consequents.
    """
    gen_to_cls = {
        tuple(itemset[1]["generators"]): (
            tuple(itemset[1]["closed_itemsets"]),
            itemset[1]["support"],
        )
        for itemset in closed_frequent_itemsets.iterrows()
    }

    generating_set = generic_basis(gen_to_cls)
    generating_set.extend(
        transitive_reduction_of_informative_basis(gen_to_cls, min_conf)
    )

    return DataFrame(generating_set, index=[i for i in range(len(generating_set))])


def generic_basis(
    generators: Dict[Tuple[str], Tuple[Tuple[str], float]]
) -> List[Dict[str, Any]]:
    """Calculates the generic basis for exact valid association rules as described in
    in 'Mining minimal non-redundant association rules'.

    Args:
        generators (Dict[Tuple[str], Tuple[Tuple[str], float]]): Mapping from generators to their closures and support

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing the antecedent and consequent as tuples, aswell as the
        support and confidence for each rule.
    """
    gb = []
    for generator, cls_info in generators.items():
        closure, supp = cls_info
        if closure != generator:
            consequent = tuple(sorted(set(closure) - set(generator)))
            row_entry = {
                "antecedents": generator,
                "consequents": consequent,
                "support": supp,
                "confidence": 1,
            }
            gb.append(row_entry)

    return gb


def transitive_reduction_of_informative_basis(
    generators: Dict[Tuple[str], Tuple[Tuple[str], float]], min_conf: float
) -> List[Dict[str, Any]]:
    """Calculates the transitive reduction of the informative basis for approximate association rules according
    to the paper 'Mining minimal non-redundant association rules'.

    Args:
        generators (Dict[Tuple[str], Tuple[Tuple[str], float]]): Mapping from generators to their closures and support.
        min_conf (float): Minimum confidence threshold.

    Returns:
        List[Dict[str, Any]]: List of dictionaries containing the antecedent and consequent as tuples, aswell as the
        support and confidence for each rule.
    """
    # Calculate the size of the longest maximal frequent closed itemset
    # and partition the FCs based on their length
    mu = 0
    FC_j = {}
    for cls, supp in generators.values():
        size_cls = len(cls)
        mu = max(size_cls, mu)
        if FC_j.get(size_cls) != None:
            FC_j[size_cls].update({cls: supp})
        else:
            FC_j[size_cls] = {cls: supp}

    ib = []
    for generator, cls_info in generators.items():
        closure, gen_supp = cls_info
        closure = set(closure)
        successors = []
        S = []  # Union of S_j

        # Determine the set of all fc_s that may be rhs of a rule
        skip = {}
        for j in range(len(closure), mu + 1):
            if FC_j.get(j) == None:
                skip[j] = True
                s_j = {}
            else:
                s_j = {fci: supp for fci,
                       supp in FC_j[j].items() if closure < set(fci)}
            S.append(s_j)

        for j in range(len(S)):
            if skip.get(j):
                continue
            for fci in S[j]:
                fci_set = set(fci)
                # Check whether there's no real subset in succ_g
                if all(not fci_set > s for s in successors):
                    successors.append(fci_set)
                    consequent = tuple(sorted(fci_set - set(generator)))
                    support_fc = FC_j[len(closure) + j][fci]
                    conf = support_fc / gen_supp

                    if conf >= min_conf:
                        ib.append(
                            {
                                "antecedents": generator,
                                "consequents": consequent,
                                "support": support_fc,
                                "confidence": conf,
                            }
                        )
    return ib


def classification_rules(
    frequent_itemsets: DataFrame, label: str, min_conf: float = 0.5
) -> DataFrame:
    """Constructs association rules from frequent itemsets directly.
    The label is expected to be a singe string which is the attribute
    of the consequent.

    Args:
        frequent_itemsets (DataFrame): Frequent itemsets to mine rules from.
        label (str): Name of the class label attribute.
        min_conf (float, optional): Minimum confidence threshold. Defaults to 0.5.

    Returns:
        DataFrame: All rules where the itemsets had the class label as an item.
        This item is placed in the consequent and the rest of the items constitutes
        the antecedent. 
    """
    # Map each item to its support
    support_mapping = frequent_itemsets.set_index(
        "itemsets").to_dict()["support"]

    # Skip over too short rules or itemset not containing the label
    frequent_itemsets = frequent_itemsets[(frequent_itemsets['itemsets'].map(len) >= 2) &
                                          (frequent_itemsets['itemsets'].map(lambda x: any(label in str(i) for i in x))) &
                                          (frequent_itemsets['support'] != 0)]

    if "ignore" in frequent_itemsets.columns:
        frequent_itemsets = frequent_itemsets[(
            frequent_itemsets["ignore"] != True)]

    rules = []
    for idx, row in frequent_itemsets.iterrows():
        itemset = row["itemsets"]
        support = row["support"]

        # Build antecedent and consequent
        rule = {}
        antecedent = []
        consequent = None
        for item in itemset:
            if label not in str(item):
                antecedent.append(item)
            else:
                consequent = (item,)

        antecedent = tuple(antecedent)
        conf = confidence(support_mapping[antecedent], support)
        if conf < min_conf:
            continue
        rule = {"antecedents": antecedent, "consequents": consequent,
                "support": support, "confidence": conf}
        rule.update(
            measure_dict(
                support_mapping[antecedent],
                support_mapping[consequent],
                support
            )
        )
        rules.append(rule)

    return DataFrame(
        rules,
        index=[i for i in range(len(rules))],
    )


def get_classification_rules(rules: DataFrame, label: str) -> DataFrame:
    """Post-Processing of rules, to only filter out rules, that have the
    classification label as the only consquent of the rule.

    Args:
        rules (DataFrame): Mined rules, superset of classification rules
        label (str): Target attribute

    Returns:
        DataFrame: All rules with only the label as consequent.
    """
    return rules.loc[
        rules["consequents"].apply(lambda x: len(
            x) == 1 and x[0].startswith(label))
    ]


def prune_by_improvement(
    db: DataFrame, rules: DataFrame, minimp: float = 0.002
) -> DataFrame:
    """Calculates the improvement for all rules and prunes any rules that do not
    fulfill the minimp constraint. It also finds all the subrules that are not conatained
    within rules to stick to the definition of improvement.

    Args:
        db (DataFrame): Database the rules were minded from
        rules (DataFrame): Mined rules
        minimp (float, optional): Minimum improvement threshold. Defaults to 0.002.

    Returns:
        DataFrame: Pruned rule set containing only productive rules.
    """
    potential_rules = _compare_to_mined_rules(rules, minimp)
    subsets = _get_proper_subsets(potential_rules)
    supports = _get_subset_supports(db, subsets)
    return _prune_by_improvement(potential_rules, supports, minimp)


def _prune_by_improvement(
    rules: DataFrame, support_info: Dict[Tuple[int], Any], minimp: float
) -> DataFrame:
    """Uses all the support information stored in support info to calculate the max confidence of any
    subrule for all the rules in the rules DataFrame.

    Args:
        rules (DataFrame): Mined rules
        support_info (Dict[Tuple[int], Any]): Information required to calculate by the improvement defintion
        minimp (float): Minimum improvement threshold

    Returns:
        DataFrame: All rules with the unproductive rules being pruned
    """
    drop_rows = []
    for idx, row in rules.iterrows():
        rule = row["antecedents"]
        items = sorted(rule)
        itemsets = list(
            chain.from_iterable(combinations(items, r)
                                for r in range(1, len(items)))
        )
        for itemset in itemsets:
            ant_sup = support_info[itemset]
            cons_sup = support_info[itemset + row["consequents"]]
            if row["confidence"] - cons_sup / ant_sup < minimp:
                drop_rows.append(idx)
                break

    return rules.drop(index=drop_rows)


def _compare_to_mined_rules(rules: DataFrame, minimp: float) -> DataFrame:
    """Checks the improvement constraint for the set of mined rules by searching for
    rules, whose antecedents are real subsets.

    Args:
        rules (DataFrame): Set of mined rules, with a single consequent
        minimp (float): Minimum improvement threshold

    Raises:
        Exception: When more than one attribute is present in the consequent an exception is raised.

    Returns:
        DataFrame: Pruned ruleset using the above condition.
    """
    if (rules["consequents"].map(len) > 1).any():
        raise Exception("Only a single attribute as antecedent allowed.")
    drop_rows = set()

    # Preprocess the DataFrame
    rules['rule_items'] = rules.apply(
        lambda x: frozenset(x['antecedents'] + x['consequents']), axis=1)
    unique_rules = rules.drop_duplicates(subset='rule_items')

    rule_lookup = dict()
    for _, row in unique_rules.iterrows():
        rule_items = row['rule_items']
        rule_conf = row['confidence']
        for key, value in rule_lookup.items():
            if key < rule_items and rule_conf - value < minimp:
                drop_rows.add(row.name)
                break
            elif rule_items < key:
                drop_rows.add(
                    unique_rules[unique_rules['rule_items'] == key].index[0])
                rule_lookup[rule_items] = rule_conf
                break
        else:
            rule_lookup[rule_items] = rule_conf

    return rules.drop(index=drop_rows).drop(labels=["rule_items"], axis=1)


def _get_proper_subsets(rules: DataFrame) -> Dict[Tuple[Any], int]:
    """Generates all proper subsets of the itemsets that make up a rule in the given set
    of potential rules (the already pruned rules do no longer have to be respected).

    Args:
        rules (DataFrame): Set of rules to get all subsets from

    Returns:
        Dict[Tuple[Any], int]: Itemsets with count 0
    """
    required_sets = set()

    # If there's any subset relation prevent that the same subsets
    # have to be recomputed
    grouped_rules = rules.groupby("consequents")
    drop_list = []
    for _, group in grouped_rules:
        for i in range(len(group)):
            row = rules.iloc[i]
            for j in range(i+1, len(group)):
                other = rules.iloc[j]
                if set(row["antecedents"]) < set(other["antecedents"]):
                    drop_list.append(row.name)
                    break

    rules = rules.drop(drop_list)

    for row in rules.itertuples(index=False, name=None):
        rule = row[0]
        items = sorted(rule)
        itemsets = set(
            chain.from_iterable(combinations(items, r)
                                for r in range(1, len(items)))
        )
        for itemset in itemsets:
            required_sets.add(itemset + row[1])
        required_sets.update(itemsets)

    return {itemset: 0 for itemset in required_sets}


def _get_subset_supports(
    db: DataFrame, subsets: Dict[Tuple[Any], int]
) -> Dict[Tuple[Any], int]:
    """Counts the support for all subsets generated by the _get_proper_subsets function.
    It thereby increments the counts associated with each itemset.

    Args:
        db (DataFrame): Database that was initially mined
        subsets (Dict[Tuple[Any], int]): All subsets with support set to 0

    Returns:
        Dict[Tuple[Any], int]: All subsets with their support in the given DB
    """
    for idx, row in db.iterrows():
        for itemset in subsets.keys():
            supported = True
            for item in itemset:
                supported = __compare_attribute(row, item)
                if supported == False:
                    break

            if supported:
                subsets[itemset] += 1

    return subsets


def __compare_attribute(row: Series, item: str) -> bool:
    """Parses the string describing an item of the itemset to get the involved attributes
    and values/interval boundaries. It then compares these informations with the
    current db row.

    Args:
        row (Series): Row of the database to match with the item
        item (str): Description of an item

    Returns:
        bool: True when the item is supported, False otherwise
    """
    # Handle clustering {x,y} = [20,30] x [25,35]
    if item.startswith("{"):
        attrlist = item[1: item.find("}")]
        names = [name.strip() for name in attrlist.split(",")]
        lower_boundaries = [
            s.strip() for s in item[item.find("[") + 1: item.find("]")].split(",")
        ]
        second_interval = item[item.find("x") + 3:]
        upper_boundaries = [
            s.strip() for s in second_interval[: second_interval.find("]")].split(",")
        ]

        for i in range(len(names)):
            name = names[i]
            if row[name] < float(lower_boundaries[i]) and row[name] > float(
                upper_boundaries[i]
            ):
                return False
        return True

    else:
        name, _, value = item.partition("=")
        name = name.strip()
        value = value.strip()
        if ".." in value:
            # Numeric attributes: x = 123..456
            lower, _, upper = value.partition("..")
            return float(lower) <= row[name] <= float(upper)
        else:
            return str(row[name]) == value
