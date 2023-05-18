# DPS (Differential Particle Solver)

DPS is a Python library for solving and analyzing differential equations in the field of particle dynamics. It provides implementations of solvers using different libraries such as Scipy, PyTorch, and JAX, allowing users to choose the most suitable solver for their specific needs.

## Table of Contents
- [Introduction](#introduction)
- [Usage](#usage)
  - [Solving ODEs](#solving-odes)
  - [Optimizing Parameters](#optimizing-parameters)
- [Contributing](#contributing)
- [License](#license)

## Introduction
DPS offers a set of tools to solve and analyze ordinary differential equations (ODEs) commonly found in particle dynamics problems. It supports two different models: **lorenz** and **guiding-center**. The **lorenz** model represents the Lorenz system of equations, while the **guiding-center** model describes the guiding-center equations.

## Usage
To use DPS, you can clone the repository and run the Python scripts `main_odes.py` and `main_diffopt.py` directly. These scripts provide different functionalities for solving ODEs and optimizing parameters, respectively.

### Solving ODEs
To solve ODEs using DPS, run the following command:
```shell
python main_odes.py
```

This script solves the ODEs based on the specified model, using different solvers such as Scipy, PyTorch, and JAX. The results will be saved in the `results` folder.

### Optimizing Parameters
To optimize parameters using DPS, run the following command:
```shell
python main_diffopt.py
```

This script optimizes the parameters of the system to achieve a specific target value for a given coordinate. It uses different solvers such as Scipy, PyTorch, and JAX. The optimization results will be saved in the `results` folder.

## Contributing
Contributions to DPS are welcome! If you encounter any issues or have suggestions for improvement, please feel free to open an issue or submit a pull request on the [DPS GitHub repository](https://github.com/rogeriojorge/DPS).

## License
DPS is released under the GNU General Public License v3.0. Please refer to the [LICENSE](https://github.com/rogeriojorge/DPS/blob/main/LICENSE) file for more details.