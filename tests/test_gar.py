import random
import pandas as pd

from algs.gar import Gene, Individuum, _amplitude, _generate_first_population, _get_fittest, _get_lower_upper_bound, _process


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
        intervals = _get_lower_upper_bound(self.data, self.description)
        result = _generate_first_population(self.data, 5, intervals)
        # Every item should at least get 2 attributes and at max 3
        assert all(2 <= len(ind.get_items()) and len(
            ind.get_items()) <= 3 for ind in result)
        # The population size has been set to 5
        assert len(result) == 5

    def test_process(self):
        self._setup()
        intervals = _get_lower_upper_bound(self.data, self.description)
        population = _generate_first_population(self.data, 5, intervals)
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
        population = _generate_first_population(self.data, 5, intervals)
        for itm in population:
            itm.fitness = random.random()
        fittest, remaining = _get_fittest(population, 0.2)
        # 0.2*5 + 1 = 2
        assert len(fittest) == 2
        assert len(remaining) == 3
        # Check that only the fittest elements were selected for the next generation
        assert all(x.fitness <= fittest[-1].fitness for x in remaining)
