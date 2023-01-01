from math import floor
import random
from typing import Any, Dict, List, Tuple
import numpy as np
import pandas as pd


class Gene:
    """Store the information associated with an individual attribute.
    For categorical attributes lower, upper is meaningless same goes for 
    numerical ones and value.
    """

    def __init__(self, name: str, numerical: bool, lower: float, upper: float, value: Any) -> None:
        self.name = name
        self.numerical = numerical
        self.upper = upper
        self.lower = lower
        self.value = value

    def is_numerical(self) -> bool:
        return self.numerical

    def __repr__(self) -> str:
        if not self.numerical:
            return f"{self.name}: {self.value}"
        else:
            return f"{self.name}: [{self.lower}, {self.upper}]"


class Individuum:

    def __init__(self, items: Dict[str,  Gene]) -> None:
        self.items = items
        self.fitness = 0.0
        self.coverage = 0
        self.marked = 0

    def num_attrs(self) -> int:
        return len(self.items)

    def get_fitness(self) -> float:
        return self.fitness

    def get_items(self) -> Dict[str, Gene]:
        return self.items

    def matches(self, record: pd.Series) -> bool:
        for name, gene in self.items.items():
            val = record[name]
            if gene.is_numerical() and (val > gene.upper or val < gene.lower):
                return False
            elif not gene.is_numerical() and (val != gene.value):
                return False

        return True

    def __repr__(self) -> str:
        return self.items.__repr__()


def _get_lower_upper_bound(db: pd.DataFrame, num_cat_attrs: Dict[str, bool]) -> Dict[str, Tuple[float, float]]:
    """Determines a dictionary where for all numerical attributes the maximum and minimum value for 
    the intervals are obtained.

    Args:
        db (pd.DataFrame): The database storing the domain information.
        num_cat_attrs (Dict[str, bool]): Mapping marking categorical and numerical attributes.

    Raises:
        Exception: When not all attributes in db given in num_cat_attrs, then this exception is raised.

    Returns:
        Dict[str, Tuple[float, float]]: Mapping from all numerical attributes to their bounding boxes [min,max].
    """
    if len(num_cat_attrs) < len(list(db.columns)):
        raise Exception(
            "Need to specify the type for each attribute in the database.")

    interval_boundaries = {}
    for name, is_num in num_cat_attrs.items():
        if is_num:
            min_val = db[name].min()
            max_val = db[name].max()
            interval_boundaries[name] = (min_val, max_val)

    return interval_boundaries


def _generate_first_population(db: pd.DataFrame, population_size: int, interval_boundaries: Dict[str, Tuple[float, float]]) -> List[Individuum]:
    """Determines an initial population, where each individuum may have 2 to n randomly sampled attributes.
    Further to come up with an individuum that is covered by at least one tuple, a random tuple from the db
    is sampled. For numeric attributes a random uniform number from 0 to 1/7 of the entire domain is added/
    subtracted from the interval boundaries.
    Note: There is no specification on how to exactly implement this in 'An Evolutionary Algorithm to Discover 
    Numeric Association Rules'.

    Args:
        db (pd.DataFrame): Database to sample initial individuals from.
        population_size (int): Number of individuals in the inital population.
        interval_boundaries (Dict[str, Tuple[float, float]]): Result of _get_lower_upper_bound

    Returns:
        List[Individuum]: Initial population.
    """
    individuums = []

    for i in range(population_size):
        item = {}
        items = list(db.columns)
        # Add two random attributes and then fill up with a coin toss for each attribute
        attrs = random.sample(items, 2)
        attrs = [itm for itm in items if itm not in attrs and random.random()
                 >= 0.5] + attrs
        row = floor(random.uniform(0, len(db)-1))
        register = db.iloc[row]

        for column in attrs:
            value = register[column]
            if interval_boundaries.get(column):
                # Add/Subtract at most 1/7th of the entire attribute domain
                lower, upper = interval_boundaries[column]
                u = floor(random.uniform(0, (upper-lower) / 7))
                lower = max(lower, value - u)
                upper = min(upper, value + u)
                item[column] = Gene(column, True, lower, upper, lower)
            else:
                value = register[column]
                item[column] = Gene(column, False, value, value, value)

        individuums.append(Individuum(item))

    return individuums


def _process(db: pd.DataFrame, marked_rows: Dict[int, bool], population: List[Individuum]) -> None:
    """Counts the number of records each individual covers aswell as the number of 
    covered records that are already marked and stores them in the individual.

    Args:
        db (pd.DataFrame): Database 
        marked_rows (Dict[int, bool]): Rows that are already covered by some fittest itemset
        population (List[Individuum]): Current population
    """
    for row in range(len(db)):
        record = db.iloc[row]
        for individual in population:
            if individual.matches(record):
                individual.coverage += 1
                individual.marked += 1 if marked_rows[row] else 0


def _amplitude(intervals: Dict[str, Tuple[float, float]], ind: Individuum) -> float:
    """Calculates the average amplitude over all numerical attributes.
    Sum over all attributes with (ind.upper - ind.lower) / (attr.upper - attr.lower) 
    divided by the number of numeric attributes.

    Args:
        intervals (Dict[str, Tuple[float, float]]): Result of _get_upper_lower_bound
        ind (Individuum): Individual whose marked and coverage fields have been set

    Returns:
        float: The average amplitude used to penalize the fitness.
    """
    avg_amp = 0.0
    count = 0
    for name, gene in ind.get_items().items():
        if intervals.get(name):
            lower, upper = intervals[name]
            avg_amp += (gene.upper - gene.lower) / (upper - lower)
            count += 1

    return avg_amp / count


def _cross_over() -> List[Individuum]:
    return []


def _mutate() -> None:
    pass


def _get_fittest() -> Individuum:
    pass


def _penalize() -> None:
    pass


def gar(db: pd.DataFrame, num_cat_attrs: Dict[str, bool], num_sets: int, num_gens: int, population_size: int, omega: float, psi: float, mu: float) -> None:
    def _get_fitness(coverage, marked, amplitude, num_attr) -> float:
        return coverage - marked*omega - amplitude*psi + num_attr*mu

    intervals = _get_lower_upper_bound(db, num_cat_attrs)
    # Store which rows of the DB were marked
    marked_rows: Dict[int, bool] = {row: False for row in range(len(db))}
    for n_itemsets in range(num_sets):
        population = _generate_first_population(db, population_size, intervals)
        for n_gen in range(num_gens):
            _process(db, marked_rows, population)

            for individual in population:
                individual.fitness = _get_fitness(individual.coverage / len(db), individual.marked/len(
                    db), _amplitude(intervals, individual), individual.num_attrs() / len(num_cat_attrs))
            _get_fittest()
    return
