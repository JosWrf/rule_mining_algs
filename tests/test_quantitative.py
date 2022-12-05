import pandas as pd

from algs.quantitative import Item, discretize_values, find_frequent_items, quantitative_itemsets

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

    def test_find_frequent_items(self):
        self._setup()
        mappings, db = discretize_values(self.data.copy(deep=True), {"age":4, "married": 0, "num_cars":0})
        result = find_frequent_items(mappings, db, {"age":4, "married": 0, "num_cars":0}, min_supp=0.4, max_supp=0.5)
        assert len(result) == 10
        assert (Item("age",0,0),) in result # 2 persons between 23-26
        assert (Item("age",1,1),) not in result # only 1 person between 27-20
        # Check whether intervals were merged 
        assert (Item("age",0,2),) in result
        assert (Item("age",1,3),) in result 

        # Both marriage values have minsupport
        assert (Item("married",0,0),) in result
        assert (Item("married",1,1),) in result

        assert (Item("num_cars",0,0),) not in result # only one person w/o car
        assert (Item("num_cars",2,2),) in result # 2 persons with 2 cars

    def test_quantitative_itemsets(self):
        self._setup()
        result = quantitative_itemsets(self.data, {"age":4, "married": 0, "num_cars":0}, minsupp=0.4, maxsupp=0.5)
        assert (Item("age",2,3), Item("married",1,1)) in result # Age: 30..38, Married: Yes (paper)
        assert (Item("age",0,0), Item("num_cars",1,1)) in result # Age: 23..26, NumCars: 1 (Support of 0.4)

        # Check whether frequent items are returned
        assert (Item("age",0,0),) in result 
        assert (Item("married",0,0),) in result
        assert (Item("num_cars",1,1),) in result 