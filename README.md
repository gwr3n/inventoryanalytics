A Python library dedicated to Inventory Analytics.

[![GitHub](https://img.shields.io/github/license/gwr3n/inventoryanalytics)](https://github.com/gwr3n/inventoryanalytics)
[![PyPI](https://img.shields.io/pypi/v/inventoryanalytics?logo=pypi)](https://pypi.org/project/inventoryanalytics)
[![PyPI - Downloads](https://img.shields.io/pypi/dm/inventoryanalytics?logo=pypi)](https://pypi.org/project/inventoryanalytics)

## Installation

Inventory Analytics supports Python 3.8 through 3.10.

```console
python -m pip install .
```

## Command line interface

The CLI provides a curated catalog of the forecasting and inventory-control
algorithms implemented by the package:

```console
inventoryanalytics -list
```

Select an algorithm with `-method`. Each algorithm exposes only its relevant
inputs; use `--help` after selecting it to see them:

```console
inventoryanalytics -method els --help
inventoryanalytics -method els \
  --n 3 \
  --p '[400, 400, 500]' \
  --d '[50, 50, 60]' \
  --h '[20, 20, 30]' \
  --s '[0.1, 0.1, 0.1]' \
  --K '[2000, 2500, 800]'
```

List and structured inputs use JSON. Results are emitted as JSON. The same
commands are available through `python -m inventoryanalytics`.

## Building

```console
python -m pip install build
python -m build
```

To cite inventoryanalytics:

```
@software{inventoryanalytics_github,
  author = {Roberto Rossi},
  title = {inventoryanalytics: a Python library dedicated to Inventory Analytics},
  url = {https://github.com/gwr3n/inventoryanalytics},
  version = {2.1},
  year = {2026}
}
```
