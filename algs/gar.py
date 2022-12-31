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

    def __repr__(self) -> str:
        if not self.numerical:
            return f"{self.name}: {self.value}"
        else:
            return f"{self.name}: [{self.lower}, {self.upper}]"


class Individuum:

    def __init__(self, items: Dict[str,  Gene]) -> None:
        self.items = items
        self.fitness = 0.0
        self.coverage = 0.0

    def get_items(self) -> Dict[str, Gene]:
        return self.items

    def get_num_attrs(self) -> int:
        return len(self.items)

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


def _get_fitness() -> float:
    return 0.0


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


def _process() -> List[Individuum]:
    return []


def _cross_over() -> List[Individuum]:
    return []


def _mutate() -> None:
    pass


def _get_fittest() -> Individuum:
    pass


def _penalize() -> None:
    pass


def gar(db: pd.DataFrame, num_cat_attrs: Dict[str, bool], num_sets: int, num_gens: int, population_size: int) -> None:
    # TODO: Implement me
    intervals = _get_lower_upper_bound(db, num_cat_attrs)
    _generate_first_population(db, population_size, intervals)
    return
