# Code Repository for Assocation Rule Mining Algorithms

In this repository different association rule mining algorithms will be implemented and tested against different datasets from several fields simulating various use cases. In the first step
a framework of the different algorithm implementations is built, which is then extended into a simple app, to compare the algorithms with different inputs and paramerters.

## Different Algorithms

- FP Tree
- Max/Closed frequent Itemset Algorithms
- Constraint-based Algorithms
- Multidimenional Algorithms

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
- Optimize the apriori algorithm. Most likely the support method rn is the bottleneck.
- Can adapt candidate generation in apriori by extending a set only with candidate k-1 itemsets when k-2 elements match and the k-1 th element of one set is greater than the any item in the other.
- Implement FP-Tree Algorithm and Variants
- Implement Constraint-based Algorithms
- Implement Multidimensional Algorithms
- Implement 
- Implemnet ARM based Classification Algorithms 
```
