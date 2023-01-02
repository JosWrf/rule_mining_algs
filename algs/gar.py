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
        genes1 = {}
        for name, genes in self.get_items().items():
            if other_genes.get(name) == None:
                genes1[name] = genes
            else:
                genes1[name] = genes if random.random(
                ) > probability else other_genes[name]

        genes2 = {}
        for name, genes in other_genes.items():
            if self.items.get(name) == None:
                genes2[name] = genes
            else:
                genes2[name] = genes if random.random(
                ) > probability else self.items[name]

        return (Individuum(genes1), Individuum(genes2))

    def mutate(self, db: pd.DataFrame, probability: float) -> None:
        """Mutates randomly selected genes. For numeric genes the interval bounaries
        are either increased or deacreased by [0, interval_width/11]. In case of 
        categorical attributes there's a 25% of changing the attribute to a random 
        value of the domain.

        Args:
            db (pd.DataFrame): Database
            probability (float): Mutation probability
        """
        for gene in self.items.values():
            # Mutate in this case
            if random.random() < probability:
                name = gene.name
                if gene.is_numerical():
                    # Change the upper and lower bound of the interval
                    lower = db[name].min()
                    upper = db[name].max()
                    width_delta = (upper - lower) / 11
                    delta1 = random.uniform(0, width_delta)
                    delta2 = random.uniform(0, width_delta)
                    gene.lower += delta1 - 1 if random.random() < 0.5 else 1
                    gene.upper += delta2 - 1 if random.random() < 0.5 else 1
                    # All this mess ensures that the interval boundaries do not exceed DB [min, max]
                    gene.lower = max(lower, gene.lower)
                    gene.upper = min(upper, gene.upper)
                    if gene.lower > gene.upper:
                        gene.upper, gene.lower = gene.lower, gene.upper
                        gene.lower = max(lower, gene.lower)
                        gene.upper = min(upper, gene.upper)

                else:
                    # Only seldomly change the value of the categorical attribute
                    gene.value = gene.value if random.random(
                    ) < 0.75 else np.random.choice(db[name].to_numpy())

    def get_all_subsets(self) -> List[Any]:
        """Generates all subsets of the current itemset.

        Returns:
            List[Any]: List of all subsets.
        """
        seeds = []
        for gene in self.items.values():
            seeds = seeds + [Individuum({gene.name: gene})] + \
                [Individuum({gene.name: gene}) + ind for ind in seeds]
        return seeds

    def to_tuple(self) -> Tuple[str]:
        items = []
        for gene in self.items.values():
            if gene.is_numerical():
                items.append(f"{gene.name} = {gene.lower}..{gene.upper}")
            else:
                items.append(f"{gene.name} = {gene.value}")
        return tuple(items)

    def __repr__(self) -> str:
        return self.items.__repr__()

    def __add__(self, other: object) -> object:
        self.items.update(other.get_items())
        return Individuum(self.items)


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
            avg_amp += (gene.upper - gene.lower) / \
                (upper - lower) if upper-lower != 0 else 1
            count += 1

    return avg_amp / count if count != 0 else 0


def _cross_over(population: List[Individuum], probability: float) -> List[Individuum]:
    """Crossover genes of the individuals and produce two offsprings, for each pair of
    randomly sampled progenitors.

    Args:
        population (List[Individuum]): Progenitors that are crossed at random
        probability (float): Crossover probability

    Returns:
        List[Individuum]: Offspring pair for each crossover event. It has double the size of 
        the given population.
    """
    recombinations = []

    for i in range(len(population)):
        progenitors = random.sample(population, k=2)
        offspring = progenitors[0].crossover(progenitors[1], probability)
        recombinations.extend(offspring)

    return recombinations


def _get_fittest(population: List[Individuum], selection_percentage: float) -> Tuple[List[Individuum], List[Individuum]]:
    """Determines the selection percentage fittest individuals and returns them as first
    element of the tuple. The other tuple element contains the remaining individuals.

    Args:
        population (List[Individuum]): Individuals of the current generation.
        selection_percentage (float): Percentage of how much individuals of the current generation pass on to the next.

    Returns:
        Tuple[List[Individuum], List[Individuum]]: Fittest individuals, Remaining ones being subject to the crossover operator
    """
    population.sort(key=lambda x: x.fitness, reverse=True)
    fittest = floor(selection_percentage*len(population) + 1)
    return (population[:fittest], population[fittest:])


def _update_marked_records(db: pd.DataFrame, marked_records: Dict[int, bool], chosen: Individuum) -> None:
    """In a postprocessing step, the itemset with the highest fitness is used to mark all the 
    records in the db, that are covered by the itemset.

    Args:
        db (pd.DataFrame): Database whose records will be marked
        marked_records (Dict[int, bool]): Stores for each record whether its already marked
        chosen (Individuum): The fittest itemset of the fully evolved population
    """
    for row in range(len(db)):
        record = db.iloc[row]
        if chosen.matches(record):
            marked_records[row] = True


def gar(db: pd.DataFrame, num_cat_attrs: Dict[str, bool], num_sets: int, num_gens: int, population_size: int,
        omega: float, psi: float, mu: float, selection_percentage: float = 0.15, recombination_probability: float = 0.5,
        mutation_probability: float = 0.4) -> pd.DataFrame:
    """Implementation of the GAR evolutionary algorithm from 'An Evolutionary Algorithm to Discover Numeric Association Rules'.
    Coverage was assumed to be relative support, amplitude was defined as (gene.upper-gene.lower) / (upper-lower), tuples were 
    marked when a chosen individual is supported by a row, a more elaborate marking could store the attributes that are covered
    for each row and use a normalized row sum. For the categorical attributes only a concrete value is stored and mutated with
    lower probability than the interval boundaries of numerical attributes.
    Note:
    Unfortunately many details of implementation were left open and some terms were not precisely defined, therefore some ideas 
    from 'An evolutionary algorithm to discover quantitative association rules from huge databases without the need for 
    an a priori discretization' were used but this again did not cover all the details.

    Args:
        db (pd.DataFrame): Database 
        num_cat_attrs (Dict[str, bool]): Maps numerical attributes to true and categorical ones to false
        num_sets (int): Number of itemsets to be generated
        num_gens (int): Number of generations
        population_size (int): Number of individuals used in each population
        omega (float): Penalization factor for coverage
        psi (float): Penalization factor for amplitude
        mu (float): Rewarding factor for attribute size
        selection_percentage (float, optional): Percentage of fittest individuals for the next generation. Defaults to 0.15.
        recombination_probability (float, optional): Probability that the offspring inherits the genes from the other progenitor. Defaults to 0.5.
        mutation_probability (float, optional): Mutation probability of numerical attributes. Defaults to 0.4.

    Returns:
        pd.DataFrame: Fittest itemsets, aswell as their subsets and support information, columns are ["itemsets","support"].
    """
    def __update_counts(db: pd.DataFrame, marked_rows: Dict[int, bool], population: List[Individuum]) -> None:
        """Processes the population and updates the coverage and marked counts.
        """
        _process(db, marked_rows, population)

        for individual in population:
            individual.fitness = _get_fitness(individual.coverage / len(db), individual.marked/len(
                db), _amplitude(intervals, individual), individual.num_attrs() / len(num_cat_attrs))

    def _get_fitness(coverage, marked, amplitude, num_attr) -> float:
        return coverage - marked*omega - amplitude*psi + num_attr*mu

    fittest_itemsets = []
    # Store which rows of the DB were marked
    marked_rows: Dict[int, bool] = {row: False for row in range(len(db))}
    intervals = _get_lower_upper_bound(db, num_cat_attrs)

    for n_itemsets in range(num_sets):
        population = _generate_first_population(db, population_size, intervals)
        for n_gen in range(num_gens):
            _process(db, marked_rows, population)

            for individual in population:
                individual.fitness = _get_fitness(individual.coverage / len(db), individual.marked/len(
                    db), _amplitude(intervals, individual), individual.num_attrs() / len(num_cat_attrs))
            next_population, remaining = _get_fittest(
                population, selection_percentage)
            offsprings = _cross_over(remaining, recombination_probability)
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

        fittest_itemsets.append(chosen_one)

    # Get all subsets of the itemsets and map into tuples to reuse the rule generation framework
    final_itemsets = []
    for itemset in fittest_itemsets:
        final_itemsets.extend(itemset.get_all_subsets())
    _process(db, marked_rows, final_itemsets)
    # Stuff into df
    final_itemsets_tuples = [{"itemsets": item.to_tuple(
    ), "support": item.coverage / len(db)} for item in final_itemsets]
    return pd.DataFrame(final_itemsets_tuples)
