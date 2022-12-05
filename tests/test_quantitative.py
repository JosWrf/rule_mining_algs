import pandas as pd

from algs.quantitative import discretize_values

class TestDiscretization:

    def _setup(self) -> None:
        # Example data in the paper Mining Quantitative Association Rules
        self.data = pd.DataFrame()
        self.data["age"] = [23,25,29,34,38]
        self.data["married"] = ["no","yes","no","yes","yes"]
        self.data["num_cars"] = [1,1,0,2,2]

    def test_partitioning(self):
        self._setup()
        mappings, db = discretize_values(self.data.copy(deep=True), {"age":4, "married": 0, "num_cars":0})
        assert len(mappings["age"]) == 4
        assert len(mappings["married"]) == 2
        assert len(mappings["num_cars"]) == 3

        # The database should only contain values that are stored as keys in the dict
        # For non interval values we can check the original db against the mapping values
        assert set(db["married"].values.flatten()) == set(mappings["married"].keys())
        assert set(self.data["married"].values.flatten()) == set(mappings["married"].values())
        assert set(db["age"].values.flatten()) == set(mappings["age"].keys())
        assert set(db["num_cars"].values.flatten()) == set(mappings["num_cars"].keys())
        assert set(self.data["num_cars"].values.flatten()) == set(mappings["num_cars"].values())