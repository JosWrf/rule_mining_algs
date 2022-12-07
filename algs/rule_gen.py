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
    support_mapping = {
        tuple(itemset[1]["itemsets"]): itemset[1]["support"]
        for itemset in frequent_itemsets.iterrows()
    }

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
            antecedent = tuple([item for item in itemset["itemsets"] if item not in consequent])
            conf = confidence(support_mapping[antecedent], support_rule)
            if conf >= min_conf:
                new_consequents.append(consequent)
                yield {
                    "antecedent": antecedent,
                    "consequent": consequent,
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
            candidate[:i] + candidate[i + 1 :] in old_candidates
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
        for j in range(len(closure), mu + 1):
            s_j = {fci: supp for fci, supp in FC_j[j].items() if closure < set(fci)}
            S.append(s_j)

        for j in range(len(S)):
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
