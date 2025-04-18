# Inference in High-Dimensional Panel Data: A Double-ML analysis of the labor market impacts of immigration at EU

## Abstract
This study focuses on estimation and inference in panel data models with
unobserved individual-specific heterogeneity in a high-dimensional context. The
framework accommodates scenarios where the number of regressors is comparable
to the sample size. Crucially, we model the individual-specific heterogeneity as fixed
effects, allowing it to correlate with observed time-varying variables in an unspecified
manner and to be non-zero for all individuals.

Within this setup, we propose methods that provide uniformly valid inference
for coefficients on a predetermined vector of endogenous variables in panel data
instrumental variables (IV) models with fixed effects and numerous instruments.
Central to the development of these methods is the application of machine learning
algorithms within a semiparametric regression framework, enabling estimation in a
grouped data structure where inter-group independence is assumed, and intragroup
dependence is unrestricted. Simulation results support the theoretical framework,
and we demonstrate the application of these methods in estimating the impact of
immigration by non-EU citizens on the employment of EU natives.

## How to create the environment to run the code

You can create your own virtual environment using the requirements.txt file. The steps are the following:

- Create a new virtual environment: 
    - `virtualenv venv`
    - `source venv/bin/activate`
    - `pip install requirements.txt`


## How to run the simulation
At the code/run_simulation_analysis.py file, you can run the simulation that compares `Pooled OLS`, `FE 2SLS`, `DML: LASSO`, `DML: XGBoost` and `DML: Random Forest`. The simulation will compare different unit sizes `(15, 50, 100, 200)` at a fixed length size `(T = 10)`. 


To run the simulation, the steps are the following:
- `python code/run_simulation_analysis.py`

## How to run the empirical analysis
At the code/run_empirical_analysis.py file, 1) data transformations are made in order to prepare the dataset for the modelling step, 2) hyperparameter tuning is made for each of the DML models and 3) the inference is calculated.

To run the empirical analysis, the steps are the following:
- `python code/run_empirical_analysis.py`
