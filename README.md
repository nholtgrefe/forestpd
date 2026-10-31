[![PyPI](https://img.shields.io/pypi/v/forestpd)](https://pypi.org/project/forestpd/)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue)](https://github.com/nholtgrefe/forestpd/blob/main/LICENSE)
[![Docs](https://img.shields.io/badge/docs-stable-blue)](https://nholtgrefe.github.io/forestpd/)

<picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://raw.githubusercontent.com/nholtgrefe/forestpd/main/docs/source/_static/forestpd-logo-dark.svg">
  <img src="https://raw.githubusercontent.com/nholtgrefe/forestpd/main/docs/source/_static/forestpd-logo.svg" alt="ForestPD logo" width="360" align="right">
</picture>

**ForestPD** (Python package `forestpd`) chooses which taxa to protect so that as much
phylogenetic diversity (PD) as possible is preserved across many phylogenetic trees at once. It uses [numba](https://numba.pydata.org/) for JIT-accelerated computation, and
connects to [phylozoo](https://github.com/nholtgrefe/phylozoo) for plotting and further analysis.

<br clear="right">

## Key Features

- **Many trees at once**: maximize PD summed over a whole collection of trees, such as posterior samples, bootstrap replicates or gene trees, optionally weighted per tree, with no summary tree or network in between.
- **Budgets and extinction risks**: assign integer costs to taxa and maximize PD under a budget; optional survival probabilities make the objective the expected PD of the surviving taxa.
- **Whole budget curves**: a single run gives the optimum, and optionally an optimal taxon set, for every budget up to the one asked for.
- **Exact, and fast when the trees mostly agree**: the running time is exponential only in the size of agreement forests of the clusters shared by all trees, so ForestPD scales to tens of thousands of taxa and thousands of trees.
- **Agreement forests**: a fast built-in heuristic, or minimum agreement forests through the optional [maffe](https://github.com/roehrt/maffe) solver.
- **Rooted and unrooted PD**: unrooted by default, rooted with `rooted=True`.
- **Interoperability**: read and write Newick, export the trees as one weighted split system (NEXUS) for split-diversity tools such as PDA, and convert to and from PhyloZoo trees.

## Installation

```bash
pip install forestpd
```

Runtime dependencies (`numpy`, `numba`) are installed automatically. For conversion to and from
PhyloZoo (e.g. for plotting), or for development:

```bash
pip install "forestpd[phylozoo]"   # PhyloZoo conversion
pip install "forestpd[dev]"        # testing
```

Minimum agreement forests need [maffe](https://github.com/roehrt/maffe), which is built from
source; see the [installation guide](https://nholtgrefe.github.io/forestpd/installation.html).

## Quickstart

Choose the taxa to protect on two trees, then add costs and survival probabilities and solve for
several budgets in one run:

```python
import forestpd as fpd

trees = ["(A:1,B:1,(C:1,D:1):2);", "(A:1,C:1,(B:1,D:1):2);"]  # or a file, one Newick tree per line

result = fpd.optimize_maxmpd(trees, budget=2, return_taxa=True)
print(result.value)         # 8.0
print(sorted(result.taxa))  # ['B', 'C']  (['A', 'D'] is optimal too)

# Costs (default 1) and survival probabilities of unprotected taxa (default 0)
instance = fpd.MaxMPDInstance(trees, costs={"A": 2, "C": 3}, survival={"B": 0.5, "D": 0.2})
budgets = [1, 2, 3, 4]
for budget, r in zip(budgets, instance.optimize_many(budgets, return_taxa=True)):
    print(budget, round(r.value, 2), sorted(r.taxa))
# 1 3.0 ['D']
# 2 6.0 ['B', 'D']
# 3 9.0 ['A', 'D']
# 4 10.0 ['A', 'B', 'D']
```

See the [user guide](https://nholtgrefe.github.io/forestpd/manual.html) for everything else,
including rooted PD, per-tree weights, pruning trees to a taxon set, and the choice of agreement
forest.

## Documentation

For the full manual, API reference, worked examples, and installation guide, visit the
**[ForestPD docs](https://nholtgrefe.github.io/forestpd/)**.

## Citation

If you use ForestPD, please cite:

> Niels Holtgrefe, Jannik Schestag, and Norbert Zeh.
> **ForestPD: Maximizing Phylogenetic Diversity under Phylogenetic and Extinction Uncertainty.**
> 2026.
