# **Machine Learning for Property Prediction of Materials**

## Overview
This is a set of Python codes based on the Scikit-learn library to predict a property for a set of given materials. Our choice of materials is functionalized MXenes, and the property of interest is their GW-level band gap.

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

