from typing import Dict, Iterator, List, Tuple
from pandas import DataFrame, Series


# TODO: Allow for more metrics and their parameters - maybe as kwargs dict
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
    ) -> Iterator[Dict[str, Tuple[Tuple[str], Tuple[str], float, float]]]:
        """Checks the minimum confidence constraint for all rules that can be built with the consequents
        in the consequents argument and yields them. The consequences are extended as long as the size is smaller than
        the size of the corresponding itemset and the frontier is not empty.

        Args:
            itemset (Series): The itemset along its support
            consequents (List[Tuple[str]]): List of all candidate consequents, which may give rules that have minimum confidence
            m (int): The size of the elements contained in consequents.

        Yields:
            Iterator[Dict[str, Tuple[Tuple[str], Tuple[str], float, float]]]: Rule[antecedent, consequent, confidence, support]
        """
        new_consequents = []
        for consequent in consequents:
            support_rule = itemset["support"]
            antecedent = tuple(sorted(set(itemset["itemsets"]) - set(consequent)))
            confidence = support_rule / support_mapping[antecedent]
            if confidence >= min_conf:
                new_consequents.append(consequent)
                yield {
                    "antecedent": antecedent,
                    "consequent": consequent,
                    "confidence": confidence,
                    "support": support_rule,
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
        index=[i for i in range(len(rules))],
        columns=["antecedent", "consequent", "confidence", "support"],
    )

    for i in range(0, len(rules)):
        df.iloc[i, 0] = rules[i]["antecedent"]
        df.iloc[i, 1] = rules[i]["consequent"]
        df.iloc[i, 2] = rules[i]["confidence"]
        df.iloc[i, 3] = rules[i]["support"]

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
