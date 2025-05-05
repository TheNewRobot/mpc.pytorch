# mpc.pytorch • [![Build Status][travis-image]][travis] [![PyPi][pypi-image]][pypi] [![License][license-image]][license]

[travis-image]: https://travis-ci.org/locuslab/mpc.pytorch.png?branch=master
[travis]: http://travis-ci.org/locuslab/mpc.pytorch

[pypi-image]: https://img.shields.io/pypi/v/mpc.svg
[pypi]: https://pypi.python.org/pypi/mpc

[license-image]: http://img.shields.io/badge/license-MIT-blue.svg?style=flat
[license]: LICENSE

*A fast and differentiable model predictive control solver for PyTorch.
Crafted by <a href="https://bamos.github.io">Brandon Amos</a>,
Ivan Jimenez,
Jacob Sacks,
<a href='https://www.cc.gatech.edu/~bboots3/'>Byron Boots</a>,
and
<a href="https://zicokolter.com">J. Zico Kolter</a>.*

---

## Installation

### Quick Setup with Python 3.8.10

```bash
# Create and activate virtual environment
python3.8 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install the package
pip install --upgrade pip
pip install mpc
```

If you want to need to un the examples in the simulator, you might need to install the specific legacy gym version
```bash
pip install gym==0.20.0 pyglet==1.2.4
```
