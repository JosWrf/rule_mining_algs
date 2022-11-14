# Code Repository for Assocation Rule Mining Algorithms

In this repository different association rule mining algorithms will be implemented and tested against different datasets from several fields simulating various use cases. In the first step
a framework of the different algorithm implementations is built.
(Maybe this framework will be embedded in a dash app if there's any time left.)

## Different Algorithms thus far

- AIS
- Apriori
- FP-Growth
- AClose
- Minimal Non-redundant rules

## Datasets
- store_data.csv: [Store data](https://user.informatik.uni-goettingen.de/~sherbold/store_data.csv)

- agaricus-lepiota.data: [Mushroom dataset](https://archive.ics.uci.edu/ml/datasets/mushroom)


## TODO

```
- Throw an exception in apriori/ais/aclose when items are not lexicographically sorted
(- Implement Constraint-based Algorithms)
- Implement Multilevel Algorithms
- Implement Multidimensional Algorithms
- Implement ARM based Classification Algorithms
```
