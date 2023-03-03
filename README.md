# Repository for Assocation Rule Mining Algorithms

Several algorithms for mining association rules have been implemented in this repository.

NOTE: The algorithms are implemented in pure Python which makes them rather slow.

## Different Algorithms thus far

- AIS
- Apriori
- FP-Growth
- h-Clique (all-confidence pushed into Apriori-algorithm)
- Quantitative Association Rules
- AClose
- Minimal Non-redundant rules
- Rule Generation for apriori-like Algorithms
- Brute-Force Mining of Classification Rules
- Clustering to find intervals for numeric attributes
- Evolutionary Algorithm to Discover Itemsets (GAR)
- Evolutionary Algorithm to Discover Rules with fixed consequent (GAR-PLUS)

## Datasets

- store_data.csv: [Store data](https://user.informatik.uni-goettingen.de/~sherbold/store_data.csv)
- agaricus-lepiota.data: [Mushroom dataset](https://archive.ics.uci.edu/ml/datasets/mushroom)

## Build and Run models

- Transformers: static_discretization(equi-depth/width partitioning of numeric attributes),
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
