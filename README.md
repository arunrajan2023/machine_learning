# **Machine Learning for Property Prediction of Materials**

## Overview
This repository contains Fortran code designed to accept DFT + NEGF-produced quantum electron transmission curves of a material (such as armchair graphene nanoribbon) at varying bias voltages (say, from -1 V to +1 V). It finds the current per the Landauer-Buttikker formalism and makes the 2D differential conductance spectrum. The two dimensions are the bias and gate voltages.
This is a set of Python codes based on Scikit-learn library to predict a propery for a set of given materials. Our choice of materials is functionalized MXenes and the property of interest is their GW-level band gap.

## Prerequisites
Before using this package, ensure the following software/files is installed/available:
- **Python**
- **Python Libraries** :
  - `Numpy`
  - `Scipy`
  - `matplotlib`
  - `scikit-learn` (library for machine learning purposes)
- **Jupyter Notebook**

## What will the script do
- Reads X (given features) and y (desired property)
- Generates primary and compound features (by applying math operations)
- Uses `LASSO` and `Random Forest regressor` to reduce/select important features
- Apply techniques such as `linear regression` and `kernel ridge regression` for property prediction

## How to run the script?

You can simply run:

```bash
python ml-2nd-level.py
```

or 
```bash
jupyter-notebook KRR-tune-hyperparam.ipynb
```

