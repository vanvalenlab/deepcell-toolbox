# ![DeepCell Toolbox Banner](https://raw.githubusercontent.com/vanvalenlab/deepcell-toolbox/master/docs/images/DeepCell_toolbox_Banner.png)

[![Build Status](https://github.com/vanvalenlab/deepcell-toolbox/workflows/build/badge.svg)](https://github.com/vanvalenlab/deepcell-toolbox/actions)
[![Coverage Status](https://coveralls.io/repos/github/vanvalenlab/deepcell-toolbox/badge.svg?branch=master)](https://coveralls.io/github/vanvalenlab/deepcell-toolbox?branch=master)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](/LICENSE)
[![PyPI version](https://badge.fury.io/py/DeepCell-Toolbox.svg)](https://badge.fury.io/py/deepcell-toolbox)
[![Python Versions](https://img.shields.io/pypi/pyversions/deepcell_toolbox.svg)](https://pypi.org/project/deepcell_toolbox/)

A collection of tools for processing data for [`deepcell-tf`](https://github.com/vanvalenlab/deepcell-tf).

# Developer instructions

First, follow the instructions for [`deepcell-tf`](https://github.com/vanvalenlab/deepcell-tf).

Then, from the `deepcell-toolbox` checkout, run:

```bash
python3.10 -m venv .venv
# Activate your environment with:
#   on Unix/macOS
#     source .venv/bin/activate
#   on Windows
#     .venv\Scripts\activate

python3.10 -m pip install --editable .

# Optionally: (to have ipython in the venv)
python3.10 -m pip install ipython
# Re-activate the venv if you installed ipython.

# Run unit tests
pytest
```
