# Programming Assignment 2

## Directory Structure

A toy neural network Python package is implemented in `toynn`. It has the following modules:

- `data.py` defines helper functions to read the MNIST data sets. The `DataSet` class implements the **shuffling** and **minibatch** functionalities.
- `layer.py` implements several common layers to be used in a network, including dense layers with **logistic**, **sigmoid** and **softmax** activation functions.
- `layer.py` implements neural network classes, one without tricks, the other one with tricks. Core algorithms such as **forward and backward propagation**, **gradient descent** and **gradient descent with momentum** are implemented here.
- `train.py` provides a helper function for training and a class to represent training result. Leaning rates **annealing** and **early stopping** is implemented here.
- `util.py` provides utilities to deal with NaN.

The Jupyter notebook `solution.ipynb` uses this `toynn` package to generate plots for our reports.

The `environment.yml` file is provided to reproduce the conda environment we use.

We assume the MNIST data is in the `data` directory.

## Reproduce Results

Python >= 3.5 and the following Python packages are required to reproduce the results:

- jupyter
- matplotlib
- numpy
- pandas
- python-mnist
- seaborn

You can install them however you like. We suggest to create a conda environment from the provided `environment.yml`

```bash
conda env create -f environment.yml
```

And activate the conda environment

```bash
source activate pa2
```

Once the requirements are satisfied, you can open Jupyter notebook

```bash
jupyter notebook
```

And run the notebook `solution.ipynb` to reproduce the results interactively. The plots will also be saved into the `figs` directory. Note that the plots won't be exactly the same as we provided in the report due to the random number generation involved in initialization. But the qualitative results shall remain the same.
