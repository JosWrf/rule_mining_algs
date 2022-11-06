# Code Repository for Assocation Rule Mining Algorithms

In this repository different association rule mining algorithms will be implemented and tested against different datasets from several fields simulating various use cases. In the first step
a framework of the different algorithm implementations is built, which is then extended into a simple app, to compare the algorithms with different inputs and paramerters.

## Different Algorithms thus far

- AIS
- Apriori
- FP-Growth

## Ideas and open problems

```
Decide what the input format will be (csv, .sqlite, .sql, .json, ...) ?
Decide on the specific algorithms.
Get data (spatial? , only transactional?, data with timestamps)
Use algorithms from libraries or still try to implement them on my own, but use libs when available and faster?
Synthetic datasets as use case?
```

## TODO

```
- Throw an exception in apriori/ais/aclose when items are not lexicographically sorted
- Implement Closed/Max frequent Itemset Algorithms
(- Implement Constraint-based Algorithms)
- Implement Multilevel Algorithms
- Implement Multidimensional Algorithms
- Implement ARM based Classification Algorithms
```
