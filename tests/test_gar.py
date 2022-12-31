import pandas as pd

from algs.gar import _generate_first_population, _get_lower_upper_bound


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
