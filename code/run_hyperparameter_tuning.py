import json
import doubleml as dml
# Main imports
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor

from run_simulations import GenerateDGP
from run_models import CausalInferenceModels

from logging_configs import setup_logging_simulation
logger = setup_logging_simulation('model_hyperparameters.log')


def hp_tuning_random_forest(dml_data):

    learner = RandomForestRegressor(n_jobs=-1)
    ml_l = clone(learner)
    ml_m = clone(learner)
    ml_r = clone(learner)

    dml_pliv_obj_rf = dml.DoubleMLPLIV(dml_data, ml_l, ml_m, ml_r)

    # Define the parameter grids for hyperparameter tuning
    par_grids_rf = {'ml_l': {'n_estimators': [50, 100, 200],
                        'max_features': [20, 50, 100, 200],
                        'max_depth': [10, 15, 20],
                        'min_samples_leaf': [1, 2, 4]},
                'ml_m': {'n_estimators': [50, 100, 200],
                        'max_features': [20, 50, 100, 200],
                        'max_depth': [10, 15, 20],
                        'min_samples_leaf': [1, 2, 4]},
                'ml_r': {'n_estimators': [50, 100, 200],
                        'max_features': [20, 50, 100, 200],
                        'max_depth': [10, 15, 20],
                        'min_samples_leaf': [1, 2, 4]}}

    # Perform hyperparameter tuning
    dml_pliv_obj_rf.tune(par_grids_rf, search_mode='grid_search')

    with open("/Users/gabrieldiasmp/Documents/pasta_gabriel/codigo/master_thesis/code/hyperparameters/" + "random_forest.json", 'w') as json_file:
        json.dump(dml_pliv_obj_rf.params, json_file, indent=4)

    return "Random Forest hyperparameter tuning successful"

def hp_tuning_xgboost(dml_data):

    # Define the parameter grids for hyperparameter tuning for XGBoost
    par_grids_xgb = {
        'ml_l': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 25],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0]
        },
        'ml_m': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 25],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0]
        },
        'ml_r': {
            'n_estimators': [50, 100, 200],
            'max_depth': [10, 20, 25],
            'learning_rate': [0.01, 0.05, 0.1],
            'subsample': [0.6, 0.8, 1.0]
        }
    }

    boost = XGBRegressor(n_jobs=-1, objective = "reg:squarederror",
                        eta=0.1, n_estimators=35)

    ml_l_boosting = clone(boost)
    ml_m_boosting = clone(boost)
    ml_r_boosting = clone(boost)

    # For boosting
    dml_pliv_obj_boosting = dml.DoubleMLPLIV(dml_data, ml_l_boosting, ml_m_boosting, ml_r_boosting)

    logger.info("tunning xgboost")
    # Perform hyperparameter tuning
    dml_pliv_obj_boosting.tune(par_grids_xgb, search_mode='grid_search')

    with open("/Users/gabrieldiasmp/Documents/pasta_gabriel/codigo/master_thesis/code/hyperparameters/" + "xgboost.json", 'w') as json_file:
        json.dump(dml_pliv_obj_boosting.params, json_file, indent=4)

    return "XGBoost hyperparameter tuning successful"

def main():
    # Simulation settings
    n_values = 50
    T = 10
    alpha = 0.5
    # num_simulations = 30

    # Instantiate DGP class
    generate_dgp = GenerateDGP(n=n_values, T=T, alpha=alpha)

    # Specify number of controls and instruments
    num_controls = int((n_values * T) * 0.5)
    num_instruments = 10

    logger.info(f"Simulation characteristics: N = {n_values} | T = {T} | num_controls = {num_controls}")
    # Generate data
    df = generate_dgp.generate_post_selection_regularization_dgp(n=n_values, px=num_controls, pz=num_instruments)

    # Extract x_columns
    x_columns = [column for column in df.columns if column.split("_")[0] == "x"]

    # Instantiate model's class
    models = CausalInferenceModels(
        df=df, y_column='y', d_columns='d', x_columns=x_columns, unit_column='unit', time_column='t', 
        desired_alpha=alpha, n=n_values
    )

    # DML setup
    dml_data = models.prepare_dml_data()

    #hp_tuning_random_forest(dml_data)

    hp_tuning_xgboost(dml_data)

    return "success!!!"

if __name__ == "__main__":
    main()
