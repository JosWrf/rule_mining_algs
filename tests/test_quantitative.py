import pandas as pd
import pytest

from algs.quantitative import (
    Item,
    _get_subintervals,
    cluster_interval_data,
    discretize_values,
    find_frequent_items,
    get_generalizations_specializations,
    quantitative_itemsets,
    _static_discretization,
    static_discretization,
)


class TestDiscretization:
    def _setup(self) -> None:
        # Example data in the paper Mining Quantitative Association Rules
        self.data = pd.DataFrame()
        self.data["age"] = [23, 25, 29, 34, 38]
        self.data["married"] = ["no", "yes", "no", "yes", "yes"]
        self.data["num_cars"] = [1, 1, 0, 2, 2]

    def test_partitioning(self):
        self._setup()
        mappings, db = discretize_values(
            self.data.copy(deep=True), {
                "age": 4, "married": 0, "num_cars": 0}, False
        )
        assert len(mappings["age"]) == 4
        assert len(mappings["married"]) == 2
        assert len(mappings["num_cars"]) == 3

        # The database should only contain values that are stored as keys in the dict
        # For non interval values we can check the original db against the mapping values
        assert set(db["married"].values.flatten()) == set(
            mappings["married"].keys())
        assert set(self.data["married"].values.flatten()) == set(
            mappings["married"].values()
        )
        assert set(db["age"].values.flatten()) == set(mappings["age"].keys())
        assert set(db["num_cars"].values.flatten()) == set(
            mappings["num_cars"].keys())
        assert set(self.data["num_cars"].values.flatten()) == set(
            mappings["num_cars"].values()
        )

    def test_find_frequent_items(self):
        self._setup()
        mappings, db = discretize_values(
            self.data.copy(deep=True), {
                "age": 4, "married": 0, "num_cars": 0}, False
        )
        result = find_frequent_items(
            mappings,
            db,
            {"age": 4, "married": 0, "num_cars": 0},
            min_supp=0.4,
            max_supp=0.5,
        )
        assert len(result) == 10
        assert (Item("age", 0, 0),) in result  # 2 persons between 23-26
        # only 1 person between 27-20
        assert (Item("age", 1, 1),) not in result
        # Check whether intervals were merged
        assert (Item("age", 0, 2),) in result
        assert (Item("age", 1, 3),) in result

        # Both marriage values have minsupport
        assert (Item("married", 0, 0),) in result
        assert (Item("married", 1, 1),) in result

        # only one person w/o car
        assert (Item("num_cars", 0, 0),) not in result
        assert (Item("num_cars", 2, 2),) in result  # 2 persons with 2 cars

    def test_quantitative_itemsets(self):
        self._setup()
        result = quantitative_itemsets(
            self.data, {"age": 4, "married": 0, "num_cars": 0}, minsupp=0.4, maxsupp=0.5
        )
        assert ("age = 31..38", "married = yes") in list(
            result["itemsets"]
        )  # As in the paper
        assert ("age = 23..26", "num_cars = 1") in list(
            result["itemsets"]
        )  # Has min supp in the table

        # Check whether frequent items are returned
        assert ("age = 23..26",) in list(result["itemsets"])
        assert ("married = no",) in list(result["itemsets"])
        assert ("num_cars = 2",) in list(result["itemsets"])

    def test_get_generalizations_specializations(self):
        frequent_itemsets = {
            (Item("a", 0, 2),): 2,
            (Item("a", 1, 2),): 1,
            (Item("a", 1, 1),): 1,
        }
        result = get_generalizations_specializations(
            frequent_itemsets, (Item("a", 0, 2),)
        )
        assert len(result[1]) == 0  # no generalizations of [0,2]
        assert {(Item("a", 1, 2),): 1, (Item("a", 1, 1),): 1} == result[
            0
        ]  # [1,2] and [1,1] are both specializations of [0,2]

        result = get_generalizations_specializations(
            frequent_itemsets, (Item("a", 1, 2),)
        )
        assert {(Item("a", 0, 2),): 2} == result[1]  # [0,2] generalizes [1,2]
        assert {(Item("a", 1, 1),): 1} == result[0]  # [0,2] specializes [1,2]

        result = get_generalizations_specializations(
            frequent_itemsets, (Item("a", 1, 1),)
        )
        assert len(result[0]) == 0  # no specializations of [1,1]
        assert {(Item("a", 1, 2),): 1, (Item("a", 0, 2),): 2} == result[
            1
        ]  # [1,1] included in [0,2] and [1,2]

    def test_get_subintervals(self):
        self._setup()
        _, db = discretize_values(self.data.copy(deep=True), {"age": 4}, False)
        itemset = (Item("age", 0, 2),)
        specializations = {(Item("age", 1, 2),): 2, (Item("age", 1, 1),): 1}
        result = _get_subintervals(db, specializations, itemset)

        assert (Item("age", 0, 1),) in result[0]  # [0,2] - [1,2] = [0,1]
        # 3 of 5 persons in age[0,1]
        assert {(Item("age", 0, 1),): 3} == result[1]
        assert (Item("age", 1, 2)) not in result[
            0
        ]  # [0,2] - [1,1] = [1,2], [0,1] so we drop it

    def test__static_discretization(self):
        self._setup()
        mappings, db = discretize_values(
            self.data.copy(deep=True), {
                "age": 4, "married": 0, "num_cars": 0}, False
        )
        result = _static_discretization(db, mappings)
        # Numer of rows should not have changed
        assert len(result) == len(db)
        # 4 (intervals for age) + 2 (married values) + 3 (num_cars values)
        assert len(result.columns) == 9

        # Each row should have 3 1's
        num_attrs = result.sum(axis=1).to_numpy()
        assert (num_attrs == 3).all(0)

        assert result.loc[0, "married = no"] == 1
        assert result.loc[0, "married = yes"] == 0
        assert result.loc[0, "age = 23..26"] == 1
        assert result.loc[0, "age = 27..30"] == 0

    def test_static_discretization(self):
        self._setup()
        discretization = {"age": 3, "married": 0, "num_cars": 0}
        result = static_discretization(self.data, discretization)
        assert len(result) == len(self.data)
        assert len(result.columns) == 8

        # Each row should have 3 1's
        num_attrs = result.sum(axis=1).to_numpy()
        assert (num_attrs == 3).all(0)

    def test_cluster_interval_data(self):
        self._setup()
        # Causes two clusters of age to be built
        attr_thresholds = {("age",): 5}
        result = cluster_interval_data(self.data, attr_thresholds)
        # Number of row should not have changed
        assert len(result) == 5

        # Number columns: 2 age clusters + 2 married values + 3 num_cars values
        assert len(result.columns) == 7

        # Check the whether the cluster inclusion was correctly encoded
        result["{age} = [23] x [29]"].values.tolist() == [True, True,
                                                          True, False, False]
        result["{age} = [34] x [38]"].values.tolist() == [False, False,
                                                          False, True, True]

        # Check if old attribute was removed
        with pytest.raises(KeyError):
            result["age"]

    def test_cluster_interval_tuple(self):
        self._setup()
        attr_thresholds = {("age", "num_cars"): 5}
        result = cluster_interval_data(self.data, attr_thresholds)
        result.drop(labels=["married = no", "married = yes"],
                    axis=1, inplace=True)

        with pytest.raises(KeyError):
            result["age"]

        with pytest.raises(KeyError):
            result["num_cars"]

        # Test presence of clustered attributes
        assert any(name.startswith("{age, num_cars}")
                   for name in result.columns)

    def test_cluster_interval_data(self):
        self._setup()
        # Causes two clusters of age to be built
        attr_thresholds = {("num_cars", "age"): 2}
        result = cluster_interval_data(self.data.drop(
            ["married"], axis=1), attr_thresholds, True)

        # There should be exactly 2 columns
        assert len(result.columns) == 2
