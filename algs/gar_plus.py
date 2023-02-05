from copy import deepcopy
import random
from math import floor
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

from algs.gar import Gene, _amplitude, _get_fittest, _get_lower_upper_bound
from algs.util import measure_dict


class RuleIndividuum:

    def __init__(self, items: Dict[str,  Gene], consequent: str) -> None:
        self.items = items
        self.consequent = consequent
        self.fitness = 0.0
        self.re_coverage = 0.0
        self.support = 0
        self.antecedent_supp = 0
        self.consequent_supp = 0
        self.attr_count = len(self.items)

    def num_attrs(self) -> int:
        return self.attr_count

    def get_fitness(self) -> float:
        return self.fitness

    def get_items(self) -> Dict[str, Gene]:
        return self.items

    def get_consequent(self) -> str:
        return self.consequent

    def confidence(self) -> float:
        if self.antecedent_supp == 0.0:
            return 0
        return self.support / self.antecedent_supp

    def reset_counts(self) -> None:
        self.fitness = 0
        self.support = 0
        self.antecedent_supp = 0
        self.re_coverage = 0

    def to_tuple(self, attrs: List[str]) -> Tuple[str]:
        items = []
        for attr in attrs:
            gene = self.items[attr]
            if gene.is_numerical():
                items.append(f"{gene.name} = {gene.lower}..{gene.upper}")
            else:
                items.append(f"{gene.name} = {gene.value}")
        return tuple(items)

    def to_dict(self, n: int) -> Dict[str, float]:
        """Converts the rule individual to an entry that is compatible with the rule
        framework in rule_gen.

        Args:
            n (int): Number of database entries.

        Returns:
            Dict[str, float]: Map with all strings s.a. antecedents, consequents, cosine, etc.
            mapped to their respective values.
        """
        antecedents = tuple(
            [item for item in self.items.keys() if item != self.consequent])
        antecedents = self.to_tuple(antecedents)
        consequent = self.to_tuple([self.consequent])

        items = {"antecedents": antecedents, "consequents": consequent,
                 "support": self.support / n,
                 "confidence": self.support / self.antecedent_supp if self.antecedent_supp != 0 else 0,
                 }
        items.update(measure_dict(self.antecedent_supp/n,
                     self.consequent_supp/n, self.support/n))

        return items

    def matching_attributes(self, record: pd.Series) -> List[str]:
        marked = []
        for name, gene in self.items.items():
            val = record[name]
            if gene.is_numerical():
                if val > gene.upper or val < gene.lower:
                    return []
            elif val != gene.value:
                return []
            marked.append(name)
        return marked

    def crossover(self, other: Any, probability: float) -> Tuple[Any, Any]:
        """Performs crossover operator to generate two offsprings from two individuals.
        Common genes are inherited by taking one at random with the given probability.
        Other genes are inherited by default.

        Args:
            other (Any): Individual to cross the current individual with
            probability (float): Crossover probability

        Returns:
            Tuple[Any, Any]: Two offsprings resulting from the crossover
        """
        other_genes = other.get_items()
        genes1 = deepcopy(self.get_items())
        genes2 = deepcopy(other_genes)

        common_genes = set(genes1).intersection(other_genes)
        rand_prob = np.random.rand(len(common_genes))
        for name, prob in zip(common_genes, rand_prob):
            if prob < probability:
                genes1[name] = deepcopy(other_genes[name])
            if prob < probability:
                genes2[name] = deepcopy(self.get_items()[name])
        return (RuleIndividuum(genes1, self.consequent), RuleIndividuum(genes2, self.consequent))

    def mutate(self, db: pd.DataFrame, probability: float) -> None:
        """Mutates randomly selected genes. For numeric genes the interval bounaries
        are either increased or deacreased by [0, interval_width/11]. In case of
        categorical attributes there's a 25% of changing the attribute to a random
        value of the domain.

        Args:
            db (pd.DataFrame): Database
            probability (float): Mutation probability
        """
        random_numbers = np.random.random(size=len(self.items))
        for i, gene in enumerate(self.items.values()):
            name = gene.name
            # Mutate in this case
            if random_numbers[i] < probability:
                name = gene.name
                if gene.is_numerical():
                    # Change the upper and lower bound of the interval
                    lower = db[name].min()
                    upper = db[name].max()
                    width_delta = (upper - lower) / 17
                    delta1 = np.random.uniform(0, width_delta)
                    delta2 = np.random.uniform(0, width_delta)
                    rands = np.random.random(size=(2))
                    gene.lower += delta1 * \
                        (-1 if rands[0] < 0.5 else 1)
                    gene.upper += delta2 * \
                        (-1 if rands[1] < 0.5 else 1)
                    # All this mess ensures that the interval boundaries do not exceed DB [min, max]
                    gene.lower = min(upper, max(gene.lower, lower))
                    gene.upper = min(upper, max(gene.upper, lower))
                    if gene.lower > gene.upper:
                        gene.upper, gene.lower = gene.lower, gene.upper

                else:
                    # Only seldomly change the value of the categorical attribute
                    gene.value = gene.value if np.random.random(
                    ) < 0.75 else np.random.choice(db[name].to_numpy())

    def __repr__(self) -> str:
        antecedent = [item.__repr__()
                      for item in self.items.values() if item.name != self.consequent]
        return f"{antecedent.__str__()} -> {self.items[self.consequent]}"


def _generate_first_rule_population(db: pd.DataFrame, population_size: int, interval_boundaries: Dict[str, Tuple[float, float]], set_attribute: str,
                                    attr_probability: float = 0.5) -> List[RuleIndividuum]:
    """Determines an initial population, where each individuum may have 2 to n randomly sampled attributes.
    Further to come up with an individuum that is covered by at least one tuple, a random tuple from the db
    is sampled. For numeric attributes a random uniform number from 0 to 1/7 of the entire domain is added/
    subtracted from the interval boundaries.

    Args:
        db (pd.DataFrame): Database to sample initial individuals from.
        population_size (int): Number of individuals in the inital population.
        interval_boundaries (Dict[str, Tuple[float, float]]): Result of _get_lower_upper_bound
        set_attribute (str): Name of attribute that should be included in every itemset.
        attr_probability (float): Probability that the attribute is not picked.

    Returns:
        List[RuleIndividuum]: Initial population.
    """
    individuums = []

    for i in range(population_size):
        item = {}
        items = list(db.columns)
        # Add two random attributes and then fill up with a coin toss for each attribute
        attrs = random.sample(items, 2)
        # If the target attribute is not sampled, removed the second sample
        if set_attribute not in attrs:
            attrs = attrs[0:1] + [set_attribute]

        attrs = [itm for itm in items if itm not in attrs and np.random.random()
                 > attr_probability] + attrs
        row = np.random.randint(0, len(db)-1)
        register = db.iloc[row]

        for column in attrs:
            value = register[column]
            if interval_boundaries.get(column):
                # Add/Subtract at most 1/7th of the entire attribute domain
                lower, upper = interval_boundaries[column]
                u = floor(np.random.uniform(0, (upper-lower) / 7))
                lower = max(lower, value - u)
                upper = min(upper, value + u)
                item[column] = Gene(column, True, lower, upper, lower)
            else:
                value = register[column]
                item[column] = Gene(column, False, value, value, value)

        individuums.append(RuleIndividuum(item, set_attribute))

    return individuums


def _count_support(db: pd.DataFrame, marked_rows: pd.DataFrame, population: List[RuleIndividuum]) -> None:
    """Updates the support count (antecedent and rule) and re-coverage for all the individuals
    in the population. Everytime a rule applies to a row, the sum of all
    the marks for that row are added and normalized by means of the row sum
    in marked_rows.

    Args:
        db (pd.DataFrame): Database to be mined
        marked_rows (pd.DataFrame): Counts how often each attribute and rule was covered
        population (List[RuleIndividuum]): Current Population
    """
    for individuum in population:
        relevant_rows = [name for name in individuum.items.keys()]
        relevant_db = pd.DataFrame(columns=[relevant_rows])

        for name, gene in individuum.items.items():
            if gene.numerical:
                relevant_db[name] = db[name].between(
                    gene.lower, gene.upper)

            else:
                relevant_db[name] = (db[name] == gene.value)

        individuum.support = relevant_db.all(axis=1).sum()
        individuum.antecedent_supp = relevant_db.drop(
            individuum.consequent, axis=1, level=0).all(axis=1).sum()

        mask = (relevant_db.all(axis=1)) & (marked_rows.sum(axis=1) != 0)
        column_sums = marked_rows.loc[mask].sum(axis=1)
        if column_sums.any():
            relevant_coverage = marked_rows[relevant_rows].loc[mask].sum(
                axis=1) / column_sums
            individuum.re_coverage = relevant_coverage.sum()


def _count_consequent_support(db: pd.DataFrame, final_rule_set: List[RuleIndividuum]) -> None:
    """Counts the support for the consequent and stores them in the respective
    RuleIndividuum.

    Args:
        db (pd.DataFrame): Database
        final_rule_set (List[RuleIndividuum]): All fittest, mined rules
    """
    for individuum in final_rule_set:
        gene = individuum.get_items()[individuum.consequent]
        mask = (db[gene.name] >= gene.lower) & (db[gene.name] <=
                                                gene.upper) if gene.is_numerical() else (db[gene.name] == gene.value)
        individuum.consequent_supp = mask.sum()


def _cross_over(population: List[RuleIndividuum], probability: float, number_offspring: int) -> List[RuleIndividuum]:
    """Crossover genes of the individuals and produce two offsprings, for each pair of
    randomly sampled progenitors.

    Args:
        population (List[RuleIndividuum]): Progenitors that are crossed at random
        probability (float): Crossover probability
        number_offspring (int): Number of remaining offsprings to generate

    Returns:
        List[RuleIndividuum]: Offspring pair for each crossover event. It has double the size of
        the given population.
    """
    recombinations = []

    for i in range(number_offspring):
        progenitors = random.sample(population, k=2)
        offspring = progenitors[0].crossover(progenitors[1], probability)
        recombinations.extend(offspring)

    return recombinations


def _update_marked_records(db: pd.DataFrame, marked_records: pd.DataFrame, chosen: RuleIndividuum) -> None:
    """In a postprocessing step, the itemset with the highest fitness is used to mark all the
    records in the db, that are covered by the itemset.

    Args:
        db (pd.DataFrame): Database whose records will be marked
        marked_records (Dict[int, bool]): Stores for each record whether its already marked
        chosen (RuleIndividuum): The fittest itemset of the fully evolved population
    """
    for i in range(len(db)):
        row = db.iloc[i]
        matches = chosen.matching_attributes(row)
        if matches:
            marked_records.loc[i, matches] += 1


def gar_plus(db: pd.DataFrame, num_cat_attrs: Dict[str, bool], num_rules: int, num_gens: int, population_size: int,
             w_s: float, w_c: float, n_a: float, w_a: float, w_recov: float, consequent: str,
             selection_percentage: float = 0.15, recombination_probability: float = 0.5, mutation_probability: float = 0.4,
             attr_probability: float = 0.5) -> pd.DataFrame:
    """Implements a version of the gar plus algorithm from 'An evolutionary algorithm to discover quantitative association rules from huge
    databases without the need for an a priori discretization', where the consequent can consist of a single item, which has to be determined
    a priori.

    Args:
        db (pd.DataFrame): Database
        num_cat_attrs (Dict[str, bool]): Maps numerical attributes to true and categorical ones to false
        num_rules (int): _description_
        num_gens (int): Number of generations
        population_size (int): Number of individuals used in each population
        w_s (float): Weighting factor for support
        w_c (float): Weighting factor for confidence
        n_a (float): Weighting factor for number attributes
        w_a (float): Weighting factor for amplitude
        w_recov (float): Weighting factor for re-coverage
        consequent (str): Consequent attribute name
        selection_percentage (float, optional): Number of individuals passing to the next generation. Defaults to 0.15.
        recombination_probability (float, optional): Crossover probability. Defaults to 0.5.
        mutation_probability (float, optional): Mutation probability. Defaults to 0.4.
        attr_probability (float, optional): Probability with which attributes are chosen, when constructing the initial 
        population. Defaults to 0.5.

    Returns:
        pd.DataFrame: Fittest rules with a bunch of measures added.
    """
    def __update_counts(db: pd.DataFrame, marked_rows: Dict[int, bool], population: List[RuleIndividuum]) -> None:
        """Processes the population and updates the coverage and marked counts.
        """
        _count_support(db, marked_rows, population)

        for individual in population:
            individual.fitness = _get_fitness(individual)

    def _get_fitness(ind: RuleIndividuum) -> float:
        result = (ind.support / n * w_s) + (ind.confidence() * w_c) + \
            (n_a * ind.num_attrs() / num_attrs) - \
            (w_a * _amplitude(intervals, individual)) - \
            (w_recov * ind.re_coverage / n)
        return result

    n = len(db)
    num_attrs = len(num_cat_attrs)

    best_rules_found = []
    intervals = _get_lower_upper_bound(db, num_cat_attrs)
    # Store a counter for each attribute, that is incremented when the row is covered by a rule
    marked_rows = pd.DataFrame(
        0, index=[i for i in range(n)], columns=list(db.columns))

    for _ in range(num_rules):
        population = _generate_first_rule_population(
            db, population_size, intervals, consequent, attr_probability)
        for n_gen in range(num_gens):
            _count_support(db, marked_rows, population)

            for individual in population:
                individual.fitness = _get_fitness(individual)

            # Get selection percentage of the best adapted individuals for the next gen
            next_population = _get_fittest(
                population, selection_percentage)
            # Crossover events two produce offspring
            offsprings = _cross_over(population, recombination_probability, len(
                population)-len(next_population))
            # Keep the better adapted of the offspring
            __update_counts(db, marked_rows, offsprings)
            offsprings = [offsprings[i] if offsprings[i].get_fitness(
            ) > offsprings[i+1].get_fitness() else offsprings[i+1] for i in range(0, len(offsprings), 2)]
            next_population.extend(offsprings)

            for individual in next_population:
                individual.mutate(db, mutation_probability)

            population = next_population

        __update_counts(db, marked_rows, population)
        chosen_one = max(population, key=lambda item: item.get_fitness())
        _update_marked_records(db, marked_rows, chosen_one)

        best_rules_found.append(chosen_one)

    # Final count of consequent support to calculate the rule measures
    _count_consequent_support(db, best_rules_found)

    # Return a dataframe containing rules only
    return pd.DataFrame([rule.to_dict(len(db)) for rule in best_rules_found]).drop_duplicates()
