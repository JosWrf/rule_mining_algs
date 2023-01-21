import random

import pandas as pd

from algs.gar import (Gene, Individuum, _amplitude, _cross_over,
                      _generate_first_population, _get_fittest,
                      _get_lower_upper_bound, _process, _update_marked_records, gar)
from algs.gar_plus import RuleIndividuum, _count_consequent_support, _count_support, _generate_first_rule_population, gar_plus
from algs.gar_plus import _update_marked_records as update_marked


class TestGar:
    def _setup(self) -> None:
        self.data = pd.DataFrame()
        self.description = {"age": True, "married": False, "temperature": True}
        self.data["age"] = [23, 25, 29, 34, 38]
        self.data["married"] = ["no", "yes", "no", "yes", "yes"]
        self.data["temperature"] = [10.5, 27.3, 40.5, -23.4, 21.96]

    def test_get_lower_upper_bound(self):
        self._setup()
        result = _get_lower_upper_bound(self.data, self.description)
        assert result == {"age": (23, 38), "temperature": (-23.4, 40.5)}

    def test_generate_first_population(self):
        self._setup()
        set_attr = "married"
        intervals = _get_lower_upper_bound(self.data, self.description)
        result = _generate_first_population(self.data, 5, intervals, set_attr)
        # Every item should at least get 2 attributes and at max 3
        assert all(2 <= len(ind.get_items()) and len(
            ind.get_items()) <= 3 for ind in result)
        # The population size has been set to 5
        assert len(result) == 5
        # Every itemset should have the married attribute
        assert all(set_attr in ind.get_items() for ind in result)

    def test_process(self):
        self._setup()
        intervals = _get_lower_upper_bound(self.data, self.description)
        population = _generate_first_population(self.data, 5, intervals, None)
        marked = {row: False for row in range(len(self.data))}
        old_coverage = sum([cov.coverage for cov in population])
        _process(self.data, marked, population)
        new_coverage = sum([cov.coverage for cov in population])
        # Every individuum is supported by at least one record
        assert old_coverage + len(self.data) - 1 < new_coverage
        # The marks should not have changed
        assert sum([1 for val in marked.values() if val]) == 0

    def test_amplitude(self):
        self._setup()
        intervals = _get_lower_upper_bound(self.data, self.description)
        genes = {"age": Gene("age", True, 27, 34, 0),
                 "married": Gene("married", False, 1, 1, 1)}
        ind = Individuum(genes)
        ind.coverage = 2
        ind.marked = 0
        result = _amplitude(intervals, ind)
        assert result == 7 / (38-23)

    def test_get_fittest(self):
        self._setup()
        intervals = _get_lower_upper_bound(self.data, self.description)
        population = _generate_first_population(self.data, 5, intervals, None)
        for itm in population:
            itm.fitness = random.random()
        fittest = _get_fittest(population, 0.2)
        # 5*0.2+1 = 2
        assert len(fittest) == 2
        # Check that only the fittest elements were selected for the next generation
        assert all(x.fitness <= fittest[-1].fitness for x in sorted(
            population, key=lambda y: y.fitness, reverse=True)[1:])

    def test_gene_crossover(self):
        genes1 = {"age": Gene("age", True, 27, 34, 30),
                  "married": Gene("married", False, 1, 1, "yes")}
        genes2 = {"age": Gene("age", True, 25, 38, 0),
                  "married": Gene("married", False, 1, 1, "yes"),
                  "temperature": Gene("temperature", True, -10, 20, 15)}
        p1 = Individuum(genes1)
        p2 = Individuum(genes2)
        result = p1.crossover(p2, 0.5)

        assert len(result) == 2
        # The first offspring has 2 attributes as its progenitor
        assert result[0].num_attrs() == 2
        # The second offspring has 3 attributes as its progenitor
        assert result[1].num_attrs() == 3
        p1_items = result[0].get_items()
        p2_items = result[1].get_items()
        # Temperature gene should stay untouched
        assert p2_items["temperature"] == genes2["temperature"]
        # Age gene is randomly chosen from either progenitor
        assert p2_items["age"] == genes1["age"] or p2_items["age"] == genes2["age"]
        assert p1_items["age"] == genes1["age"] or p1_items["age"] == genes2["age"]

    def test_crossover_types(self):
        self._setup()
        intervals = _get_lower_upper_bound(self.data, self.description)
        population = _generate_first_population(self.data, 5, intervals, None)
        num_remaining = 2
        result = _cross_over(population, 0.5, num_remaining)
        # Every item should be an idividuum and there should be 2 offsprings
        # for every element in the population
        assert all(type(x) == Individuum for x in result)
        assert len(result) == 2*num_remaining

    def test_mutate(self):
        self._setup()
        genes = {"age": Gene("age", True, 27, 34, 30),
                 "married": Gene("married", False, 1, 1, "yes")}
        ind = Individuum(genes)
        ind.mutate(self.data, 0.5)
        assert ind.num_attrs() == 2
        assert ind.get_items()[
            "married"].value in self.data["married"].to_numpy().tolist()

    def test_update_marked_records(self):
        self._setup()
        genes = {"temperature": Gene("temperature", True, 0, 25, 30)}
        ind = Individuum(genes)
        marked = {row: False for row in range(len(self.data))}
        _update_marked_records(self.data, marked, ind)
        new_marked = {0: True, 1: False, 2: False, 3: False, 4: True}
        assert marked == new_marked

    def test_get_all_subsets(self):
        genes2 = {"age": Gene("age", True, 25, 38, 0),
                  "married": Gene("married", False, 1, 1, "yes"),
                  "temperature": Gene("temperature", True, -10, 20, 15)}
        ind = Individuum(genes2)
        result = ind.get_all_subsets()
        # All subsets but the empty subset
        assert len(result) == 7
        assert max(itemset.num_attrs() for itemset in result) == 3

        genes1 = {"age": Gene("age", True, 25, 38, 0),
                  "married": Gene("married", False, 1, 1, "yes")}
        ind = Individuum(genes1)
        result = ind.get_all_subsets()
        assert len(result) == 3
        assert max(itemset.num_attrs() for itemset in result) == 2

    def test_to_tuple(self):
        genes = {"temperature": Gene(
            "temperature", True, -10, 25, 7), "age": Gene("age", True, 25, 38, 27)}
        ind = Individuum(genes)
        result = ind.to_tuple()
        assert result == ("temperature = -10..25", "age = 25..38")

    def test_gar(self):
        self._setup()
        result = gar(self.data, self.description, 3, 10, 3, 0.5, 0.4, 0.3)

        assert result["support"].min() >= 0
        assert result["support"].max() <= 1
        assert "itemsets" in list(result.columns)


class TestGarPlus:
    def _setup(self) -> None:
        self.data = pd.DataFrame()
        self.description = {"age": True, "married": False, "temperature": True}
        self.data["age"] = [23, 25, 29, 34, 38]
        self.data["married"] = ["no", "yes", "no", "yes", "yes"]
        self.data["temperature"] = [10.5, 27.3, 40.5, -23.4, 21.96]
        self.intervals = _get_lower_upper_bound(self.data, self.description)
        self.consequent = "married"

    def _generate_individuals(self) -> None:
        genes1 = {"age": Gene("age", True, 25, 38, 0),
                  "married": Gene("married", False, 1, 1, "yes")}
        genes2 = {"age": Gene("age", True, 25, 38, 0),
                  "married": Gene("married", False, 1, 1, "yes"),
                  "temperature": Gene("temperature", True, -10, 30, 15)}
        self.ind = RuleIndividuum(genes1, self.consequent)
        self.ind2 = RuleIndividuum(genes2, self.consequent)
        self.population = [self.ind, self.ind2]

    def test_generate_first_rule_population(self):
        self._setup()
        result = _generate_first_rule_population(
            self.data, 5, self.intervals, self.consequent)
        # The support and coverage should all be 0
        assert all([cov.re_coverage for cov in result]) == 0
        assert all([cov.support for cov in result]) == 0
        assert all([cov.antecedent_supp for cov in result]) == 0
        # Every item should at least get 2 attributes and at max 3
        assert all(2 <= len(ind.get_items()) and len(
            ind.get_items()) <= 3 for ind in result)
        # The population size has been set to 5
        assert len(result) == 5
        # Every itemset should have the married attribute
        assert all(self.consequent in ind.get_items() for ind in result)
        assert all(self.consequent == ind.get_consequent() for ind in result)

    def test_count_support(self):
        self._setup()
        self._generate_individuals()
        marked = pd.DataFrame(
            0, index=[i for i in range(len(self.data))], columns=list(self.data.columns))
        _count_support(self.data, marked, self.population)
        # The marks should not have changed
        assert all([cov.re_coverage for cov in self.population]) == 0
        assert marked.sum().sum() == 0

        assert self.ind.antecedent_supp == 4
        assert self.ind.support == 3

        assert self.ind2.antecedent_supp == 2
        assert self.ind2.support == 2

    def test_update_marked_records(self):
        self._setup()
        self._generate_individuals()
        marked = pd.DataFrame(
            0, index=[i for i in range(len(self.data))], columns=list(self.data.columns))
        update_marked(self.data, marked, self.ind)
        # Rule applies 3 times
        assert marked["age"].sum(axis=0) == 3
        assert marked["temperature"].sum(axis=0) == 0
        assert marked["married"].sum(axis=0) == 3

    def test_consequent_support(self):
        self._setup()
        self._generate_individuals()
        _count_consequent_support(self.data, self.population)
        assert self.ind.consequent_supp == 3
        assert self.ind2.consequent_supp == 3

    def test_to_dict(self):
        self._setup()
        self._generate_individuals()
        marked = pd.DataFrame(
            0, index=[i for i in range(len(self.data))], columns=list(self.data.columns))
        _count_support(self.data, marked, self.population)
        result = self.ind2.to_dict(len(self.data))
        assert result["antecedents"]
        # 2 instances cover the db, where the antecedents cooccur with the consequent
        assert result["antecedents"] == (
            "age = 25..38", "temperature = -10..30")
        assert result["support"] == 0.4
        assert result["confidence"] == 1

        result = self.ind.to_dict(len(self.data))
        assert result["antecedents"] == ("age = 25..38",)
        assert result["consequents"] == ("married = yes",)
        # antecedent matches 4 times and the entire itemset 3 times
        assert result["support"] == 0.6
        assert result["confidence"] == 3/4

    def test_gar_plus(self):
        self._setup()
        result = gar_plus(self.data, self.description, 3, 10,
                          3, 0.5, 0.4, 0.3, 0.2, 0.4, self.consequent)

        assert result["support"].min() >= 0
        assert result["support"].max() <= 1
        assert {"support", "confidence", "antecedents",
                "consequents", "imbalance_ratio"} < set(list(result.columns))
