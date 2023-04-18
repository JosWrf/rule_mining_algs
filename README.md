# Repository for Assocation Rule Mining Algorithms

Several algorithms for mining association rules have been implemented in this repository.

NOTE: The algorithms are implemented in pure Python which makes them rather slow.

## Installation

```bash
pip install rule-mining-algs
```

## Different Algorithms thus far

- AIS [1]
- Apriori [1]
- FP-Growth [2]
- h-Clique (all-confidence pushed into Apriori-algorithm) [3]
- Quantitative Association Rules [4]
- AClose [5]
- Minimal Non-redundant rules [6]
- Rule Generation for apriori-like Algorithms [1]
- Clustering to find intervals for numeric attributes [7]
- Evolutionary Algorithm to Discover Itemsets (GAR) [8]
- Evolutionary Algorithm to Discover Rules with fixed consequent (GAR-PLUS) [9]

## Datasets

- store_data.csv: [Store data](https://user.informatik.uni-goettingen.de/~sherbold/store_data.csv)
- agaricus-lepiota.data: [Mushroom dataset](https://archive.ics.uci.edu/ml/datasets/mushroom)

# Usage
## Models 
Models encapsulate a discretization algorithm, an itemset mining algorithm and an algorithm to generate association rules.
(In case of using the GAR and GAR-plus models no discretization is required.)

There's no need to use a model, however. 
Instead the individual algorithms can be used as is.
Models just provide an abstraction mechanism.
## Build and Run models
- Discretization Algorithms (Transformers): static_discretization(equi-depth/width partitioning of numeric attributes),
  cluster_interval_data(birch clustering to find intervals for numeric attributes)
- Itemset_Miners: see [Algorithms](#Different-Algorithms-thus-far) section
- Rule_Miners: generate_rules(Standard algorithm to generate rules from itemsets),
  min_redundant_rules(only usable with a_close itemset miner)

---

**Example for a dataset with a single attribute**

```python
from algs.models import StandardMiner
from algs.data import load_store_data

data_df = load_store_data() # Load store dataset
# Choose model
m = StandardMiner()
# Set parameters for the algorithms
m.set_args(m.itemset_miner, {"min_support": 0.005})
m.set_args(m.rule_miner, {"min_conf": 0.5})
# Run the algorithm on the dataset
output = m.run(data_df)
```

**Example for a DB containing several categorical attributes**

```python
from algs.data import load_shroom_data
from algs.quantitative import static_discretization
from algs.rule_gen import get_classification_rules

shrooms = load_shroom_data()
mine_quant = StandardMiner(static_discretization)
names = {name: 0 for name in shrooms.columns}
# Set arguments for transformer, itemset and rule miner
mine_quant.set_args(mine_quant.transformer, {"discretization": names})
mine_quant.set_args(mine_quant.itemset_miner, {"min_support": 0.15})
mine_quant.set_args(mine_quant.rule_miner, {"min_conf": 0.65})
rules = mine_quant.run(shrooms)
# Post processing step to obtain rules having only the label in the consequent
classification_rules = get_classification_rules(rules, "label")
```

## References
[1] Rakesh Agrawal and Ramakrishnan Srikant. 1994. Fast Algorithms for Mining Association Rules in Large Databases. In Proceedings of the 20th International Conference on Very Large Data Bases (VLDB '94), 487–499.

[2] Jiawei Han, Jian Pei, and Yiwen Yin. 2000. Mining frequent patterns without candidate generation. SIGMOD Rec. 29, 2 (June 2000), 1–12. 

[3] H. Xiong, P. . -N. Tan and Vipin Kumar, Mining strong affinity association patterns in data sets with skewed support distribution, Third IEEE International Conference on Data Mining, 2003, pp. 387-394.

[4] Ramakrishnan Srikant and Rakesh Agrawal. 1996. Mining quantitative association rules in large relational tables. SIGMOD Rec. 25, 2 (June 1996), 1–12. 

[5] Pasquier, N., Bastide, Y., Taouil, R., Lakhal, L. (1999). Discovering Frequent Closed Itemsets for Association Rules. In: Beeri, C., Buneman, P. (eds) Database Theory — ICDT’99. ICDT 1999. Lecture Notes in Computer Science, vol 1540. 

[6] Bastide, Yves, et al. Mining minimal non-redundant association rules using frequent closed itemsets. Computational Logic—CL 2000: First International Conference London, July 24–28, 2000 Proceedings.

[7] Miller, Renée J., and Yuping Yang. Association rules over interval data. ACM SIGMOD Record 26.2 (1997): 452-461.

[8] Mata, Jacinto, José-Luis Alvarez, and José-Cristobal Riquelme. An evolutionary algorithm to discover numeric association rules. Proceedings of the 2002 ACM symposium on Applied computing. 2002.

[9] Alvarez, Victoria Pachon, and Jacinto Mata Vazquez. An evolutionary algorithm to discover quantitative association rules from huge databases without the need for an a priori discretization. Expert Systems with Applications 39.1 (2012): 585-593.