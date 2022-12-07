import pandas as pd

from algs.quantitative import Item, _get_subintervals, discretize_values, find_frequent_items, get_generalizations_specializations, quantitative_itemsets

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
        assert ('age = 31..38', 'married = yes') in list(result["itemsets"]) # As in the paper
        assert ('age = 23..26', 'num_cars = 1') in list(result["itemsets"]) # Has min supp in the table

        # Check whether frequent items are returned
        assert ('age = 23..26',) in list(result["itemsets"]) 
        assert ('married = no',) in list(result["itemsets"]) 
        assert ('num_cars = 2',) in list(result["itemsets"]) 

    def test_get_generalizations_specializations(self):
        frequent_itemsets = {(Item("a",0,2),): 2, (Item("a",1,2),): 1, (Item("a",1,1),):1}
        result = get_generalizations_specializations(frequent_itemsets, (Item("a",0,2),))
        assert len(result[1]) == 0 # no generalizations of [0,2]
        assert {(Item("a",1,2),): 1, (Item("a",1,1),):1} == result[0] # [1,2] and [1,1] are both specializations of [0,2]

        result = get_generalizations_specializations(frequent_itemsets, (Item("a",1,2),))
        assert {(Item("a", 0, 2),): 2} == result[1] # [0,2] generalizes [1,2]
        assert {(Item("a", 1, 1),): 1} == result[0] # [0,2] specializes [1,2]
        
        result = get_generalizations_specializations(frequent_itemsets, (Item("a",1,1),))
        assert len(result[0]) == 0 # no specializations of [1,1]
        assert {(Item("a",1,2),): 1, (Item("a",0,2),):2} == result[1] # [1,1] included in [0,2] and [1,2]

    def test_get_subintervals(self):
        self._setup()
        _, db = discretize_values(self.data.copy(deep=True), {"age":4})
        itemset = (Item("age",0,2),)
        specializations = {(Item("age",1,2),): 2, (Item("age",1,1),):1}
        result = _get_subintervals(db, specializations, itemset)

        assert (Item("age", 0, 1),) in result[0] # [0,2] - [1,2] = [0,1]
        assert {(Item("age", 0, 1),): 3} == result[1] # 3 of 5 persons in age[0,1]
        assert (Item("age", 1, 2)) not in result[0] # [0,2] - [1,1] = [1,2], [0,1] so we drop it
