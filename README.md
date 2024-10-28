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

Assuming that you have the necessary input files (features and the desired property for several materials),

you can simply run:

```bash
python ml-2nd-level.py
```

or 
```bash
jupyter-notebook KRR-tune-hyperparam.ipynb
```

## Additional Files and Regression Attempts
The notebook `GBR-85x12-100_50_0.1.ipynb` also has examples on
- `SupportVectorMachine Regressor` with `GridSearchCV`
- `AdaBoostRegressor`
- `GradientBoostingRegressor`
  
We can run this notebook by
```bash
jupyter-notebook GBR-85x12-100_50_0.1.ipynb
```

## More information
More details about machine learning based property prediction can be found in our paper: [A. C. Rajan et al. Chem. Mater. 2018](https://pubs.acs.org/doi/abs/10.1021/acs.chemmater.8b00686).

## License
&copy; Arun Rajan @2018
