# forestpd documentation

Sphinx site for the **forestpd** Python package (ForestPD: maximizing phylogenetic diversity
across many trees), published at <https://nholtgrefe.github.io/forestpd/>.

## Structure

| Path | Purpose |
|------|---------|
| `source/index.rst` | Landing page: what ForestPD does, a first example, citation |
| `source/installation.rst` | Installing forestpd, the extras, maffe, troubleshooting |
| `source/problem.rst` | The MaxMPD problem and, in broad strokes, how ForestPD solves it |
| `source/manual.rst` | User guide: input, instances, budgets and curves, rooted PD, pruning, agreement forests, export |
| `source/examples.rst` | A complete analysis of a small data set, with the budget-curve figure |
| `source/api/` | One page per part of the public API (the names listed under "Public API" in the README) |
| `source/_static/` | Logos (light and dark) and the two-tree figure of the problem page |

The `>>>` examples in the pages are doctests: `pytest` runs them with the rest of the suite. The
budget-curve figure on the examples page is drawn by matplotlib while the site is built.

## Build

From the repository root:

```bash
pip install -e ".[docs]"
sphinx-build -b html docs/source docs/build/html
```

Open `docs/build/html/index.html`. Use `-W` to treat warnings as errors, as the deploy workflow
(`.github/workflows/docs.yml`, on `v*` tags or by hand) does:

```bash
sphinx-build -b html docs/source docs/build/html -W
```
