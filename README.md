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

### Quick Setup with Python 3.8.10 or greater

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
pip install gymnasium==1.1.1 "gymnasium[classic_control]" matplotlib
```

If you are running in Ubuntu 
``` bash
# For interactive plotting
sudo apt install python3-tk

# For LaTeX support in matplotlib
sudo apt install texlive texlive-latex-extra dvipng cm-super
```
## Pendulum Parameter Learning with Differentiable MPC
Complete workflow for learning pendulum dynamics using differentiable MPC through expert demonstrations.

### Quick Start
Run experiments in order:

``` bash
# 1. Baseline pendulum control
python simple_pendulum.py

# 2. Learn parameters from expert demonstrations (~30 min)
python learning_pendulum.py

# 3. Generate training plots
python plot_training_results.py

# 4. Validate learned parameters and visualize an experiment
python post_training_pendulum.py

```
### Expected Results
- Parameter Recovery: True `[g=10.0, m=1.0, l=1.0]` → Learned `[~10.1, ~0.87, ~1.09]`
- Performance: Both models successfully perform swing-up task
- Outputs: Training logs, parameter evolution plots, comparison videos

All results saved in `pendulum_experiments/` with timestamped directories.