from pathlib import Path
import numpy as np
import pandas as pd
import random 
import json

import doubleml as dml

# Main imports
from sklearn.base import clone
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.linear_model import LinearRegression, Lasso
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from xgboost import XGBRegressor

from linearmodels.iv import IV2SLS

from utils.logging_configs import setup_logging
logger = setup_logging('run_models.log')


REPO_DIR = Path(__file__).parent.parent.parent


class CausalInferenceModels:
    def __init__(
            self, df: pd.DataFrame, 
            y_column: str,  
            d_columns: list, 
            unit_column: str,
            time_column: str,
            z_columns: list=None,
            x_columns: list=None,
            n: int = None,
            desired_alpha: float=None,
            hp_params: json=None): #z_columns: list
        self.df = df
        self.y = y_column
        self.x = x_columns
        self.logger = logger
        if z_columns != None:
            self.z = z_columns
        else:
            self.z = [column for column in self.df.columns if column.split("_")[0] == "instrument"]

        self.d = d_columns
        self.n = n
        self.unit_column = unit_column
        self.time_column = time_column
        self.alpha = desired_alpha
        self.hp_params = hp_params

    def generate_point_inference(self, fit_obj, model_name, framework):
        self.actual_fitted_model = fit_obj

        def generate_results_for_single_endogenous_variable(fit_obj, framework):
            if framework == "doubleml":
                # Extract results for the endogenous variable 'D'
                results = fit_obj.summary.coef.values[0] #coef
                std_err = fit_obj.summary.iloc[:, 1].values[0]
                t_stat = fit_obj.summary.iloc[:, 2].values[0]
                p_value = fit_obj.summary.iloc[:, 3].values[0]
                ci_lower, ci_upper = fit_obj.summary.iloc[:, 4].values[0], fit_obj.summary.iloc[:, 5].values[0]
            elif framework == "statsmodels":
                # Extract results for the endogenous variable 'D'
                results = fit_obj.params.loc[self.d]
                std_err = fit_obj.std_errors.loc[self.d]
                t_stat = fit_obj.tstats.loc[self.d]
                p_value = fit_obj.pvalues.loc[self.d]
                ci_lower, ci_upper = fit_obj.conf_int().loc[self.d]

            stats_df = {
                'Coefficient': np.round(results, 4),
                'Bias': np.round(results - self.alpha, 4),
                'Standard Error': np.round(std_err, 4),
                't-Statistic': t_stat,
                'p-Value': p_value,
                '95% CI Lower': np.round(ci_lower, 4),
                '95% CI Upper': np.round(ci_upper, 4),
                'model_name': model_name,
                'size_panel': self.n
            }
    
            return stats_df

        def generate_results_for_multiple_endogenous_variables(fit_obj, framework):
            if framework == "doubleml":
                # Extract results for the endogenous variable 'D'
                results = fit_obj.summary.coef.to_dict()
                std_err = fit_obj.summary.iloc[:, 1].to_dict()
                t_stat = fit_obj.summary.iloc[:, 2].to_dict()
                p_value = fit_obj.summary.iloc[:, 3].to_dict()
                ci_lower, ci_upper = fit_obj.summary.iloc[:, 4].to_dict(), fit_obj.summary.iloc[:, 5].to_dict()
            elif framework == "statsmodels":
                # Extract results for the endogenous variable 'D'
                results = fit_obj.params.loc[self.d]
                std_err = fit_obj.std_errors.loc[self.d]
                t_stat = fit_obj.tstats.loc[self.d]
                p_value = fit_obj.pvalues.loc[self.d]
                ci_lower, ci_upper = fit_obj.conf_int().loc[self.d]

            # Create a DataFrame
            stats_df = pd.DataFrame({
                'Endogenous': list(results.keys()),
                'Coefficient': list(results.values()),
                'Standard Error': list(std_err.values()),
                't-Statistic': list(t_stat.values()),
                'p-value': list(p_value.values()),
                '95_lower_ci':list(ci_lower.values()),
                '95_upper_ci':list(ci_upper.values()),
                'model': model_name
            })
    
            return stats_df

        if framework == "statsmodels":
            results_df = generate_results_for_single_endogenous_variable(fit_obj, framework)
        elif len(fit_obj.summary.coef.tolist()) == 1:
            results_df = generate_results_for_single_endogenous_variable(fit_obj, framework)
        elif len(fit_obj.summary.coef.tolist()) > 1:
            results_df = generate_results_for_multiple_endogenous_variables(fit_obj, framework)

        return results_df

    def generate_rmse_df(self, nuisance_loss_dict, model):
        df = {
            'rmse_g(x)': nuisance_loss_dict['ml_l'][0][0],
            'rmse_m(x)': nuisance_loss_dict['ml_r'][0][0],
            'model': model  
        }

        return df

    def pooled_2sls(self):
        df_2sls = self.df.copy()
        # Convert to panel data format
        df_2sls.set_index([self.unit_column, self.time_column], inplace=True)

        #z_columns = [column for column in df_diff.columns if column.split("_")[0] == "instrument"]
        # Prepare the dependent and independent variables, and instruments

        if self.x == None:
            x_columns = self.d
        else:
            print("x_cols is present")
            x_columns = [self.d] + self.x


        Y = df_2sls[[self.y]]
        X = df_2sls[x_columns]
        Z = df_2sls[self.z]  # Include other instruments if available

        self.logger.info("Run Pooled 2SLS")

        # Perform the IV regression with clustered standard errors
        iv_model = IV2SLS(Y, X, None, Z).fit(cov_type='clustered', clusters=df_2sls.index.get_level_values(self.unit_column))

        results_df = self.generate_point_inference(iv_model, "Pooled 2SLS", "statsmodels")

        return results_df
    
    def first_difference_2sls(self):
        df_2sls = self.df.copy()
        # Convert to panel data format
        df_2sls.set_index([self.unit_column, self.time_column], inplace=True)

        # Compute first differences for all variables
        df_diff = df_2sls.groupby(level=self.unit_column).diff().dropna()

        if self.x == None:
            x_columns = self.d
        else:
            print("x_cols is present")
            x_columns = [self.d] + self.x


        Y = df_diff[[self.y]]
        X = df_diff[x_columns]
        Z = df_diff[self.z]  # Include other instruments if available

        self.logger.info("Run FD 2SLS")
        # Perform the IV regression with clustered standard errors
        iv_model = IV2SLS(Y, X, None, Z).fit(cov_type='clustered', clusters=df_diff.index.get_level_values(self.unit_column))

        results_df = self.generate_point_inference(iv_model, "First Differences", "statsmodels")

        return results_df
    
    def prepare_dml_data(self) -> dml.DoubleMLClusterData:
        # Generate dummy columns using pd.get_dummies()
        dummy_columns = pd.get_dummies(self.df[self.unit_column], prefix='individual', dtype=int)

        # Concatenate the dummy columns with the original DataFrame
        df_with_dummies = pd.concat([self.df, dummy_columns], axis=1)

        # Concatenate the dummy columns with the original DataFrame
        dummy_columns = pd.get_dummies(self.df[self.time_column], prefix='time', dtype=int)
        df_with_dummies = pd.concat([df_with_dummies, dummy_columns], axis=1)
        df_with_dummies

        individual_columns = [column for column in df_with_dummies.columns if column.split("_")[0] == "individual"]
        time_columns = [column for column in df_with_dummies.columns if column.split("_")[0] == "time"]
        instrument_columns = [column for column in df_with_dummies.columns if column.split("_")[0] == "instrument"]

        # Compute first differences for all variables
        df_with_dummies.set_index([self.unit_column, self.time_column], inplace=True)
        df_with_diff = df_with_dummies.groupby(level=self.unit_column).diff().dropna()
        #df_with_diff = df_with_dummies

        # # Apply the diff() operation to the selected columns
        # df_diff = data_filtered[columns_to_diff].groupby(level='country').diff()
        # Group by country and apply shift to create lagged columns
        df_with_diff[f'{self.y}_lag'] = df_with_diff.groupby(self.unit_column)[self.y].shift(1)
        df_with_diff[f'{self.d}_lag'] = df_with_diff.groupby(self.unit_column)[self.d].shift(1)

        lag_variables = [f'{self.y}_lag', f'{self.d}_lag']

        if self.x == None:
            x_columns = individual_columns+time_columns+lag_variables
        else:
            x_columns = self.x+individual_columns+time_columns+lag_variables

        df_with_diff = df_with_diff.dropna().reset_index()
    
        obj_dml_data = dml.DoubleMLClusterData(
            df_with_diff, y_col=self.y, d_cols=self.d, 
            x_cols=x_columns, #+time_columns 
            z_cols=instrument_columns, cluster_cols=self.unit_column)
         
        return obj_dml_data
    

    def dml_lasso(self, obj_dml_data):

        if self.hp_params:
            logger.info("Reading LASSO hyperparameters")
            instrument_key = list(self.hp_params["lasso"].keys())[2]
            endogenous_key = list(self.hp_params["lasso"]['ml_l'].keys())[0]

            # Extract hyperparameters for ml_l, ml_m, and ml_r
            ml_l_params = self.hp_params["lasso"]['ml_l'][endogenous_key][0][0]
            ml_r_params = self.hp_params["lasso"]['ml_r'][endogenous_key][0][0]
            ml_m_params = self.hp_params["lasso"][instrument_key][endogenous_key][0][0]  # Adjust this key based on the desired instrument

            def create_lasso_regressor(params):
                return make_pipeline(
                    StandardScaler(), 
                    Lasso(alpha=params["lasso__alpha"], max_iter=10000, random_state = random.seed(1234)))

            # Create the RandomForestRegressor objects
            ml_l_lasso = create_lasso_regressor(ml_l_params)
            ml_m_lasso = create_lasso_regressor(ml_m_params)
            ml_r_lasso = create_lasso_regressor(ml_r_params)

        else:
            learner_lasso = make_pipeline(StandardScaler(), Lasso(alpha=0.05, max_iter=1000, random_state = random.seed(1234)))

            ml_l_lasso = clone(learner_lasso)
            ml_m_lasso = clone(learner_lasso)
            ml_r_lasso = clone(learner_lasso)


        # For Lasso
        dml_pliv_obj_lasso = dml.DoubleMLPLIV(obj_dml_data, ml_l_lasso, ml_m_lasso, ml_r_lasso)
        
        self.logger.info("Run LASSO")        
        lasso_fit = dml_pliv_obj_lasso.fit()

        results_df = self.generate_point_inference(fit_obj=lasso_fit, model_name="DML: LASSO", framework="doubleml")

        rmse_lasso = self.generate_rmse_df(dml_pliv_obj_lasso.evaluate_learners(), "DML: LASSO")

        return results_df, rmse_lasso # Display the DataFrame

    def dml_xgboost(self, obj_dml_data):

        if self.hp_params:
            instrument_key = list(self.hp_params["xgboost"].keys())[2]
            endogenous_key = list(self.hp_params["xgboost"]['ml_l'].keys())[0]

            # Extract hyperparameters for ml_l, ml_m, and ml_r
            ml_l_params = self.hp_params["xgboost"]['ml_l'][endogenous_key][0][0]
            ml_r_params = self.hp_params["xgboost"]['ml_r'][endogenous_key][0][0]
            ml_m_params = self.hp_params["xgboost"][instrument_key][endogenous_key][0][0]  # Adjust this key based on the desired instrument

            def create_xgboost_regressor(params):
                return XGBRegressor(
                    learning_rate=params['learning_rate'],
                    max_depth=params['max_depth'],
                    n_estimators=params['n_estimators'],
                    n_jobs=-1  # Use all available cores
                )

            # Create the RandomForestRegressor objects
            ml_l_boosting = create_xgboost_regressor(ml_l_params)
            ml_m_boosting = create_xgboost_regressor(ml_m_params)
            ml_r_boosting = create_xgboost_regressor(ml_r_params)

        else:
            learner_boosting = XGBRegressor(n_jobs=-1, objective = "reg:squarederror",
                                eta=0.1, n_estimators=50)
            
            ml_l_boosting = clone(learner_boosting)
            ml_m_boosting = clone(learner_boosting)
            ml_r_boosting = clone(learner_boosting)


        # For boosting
        dml_pliv_obj_boosting = dml.DoubleMLPLIV(obj_dml_data, ml_l_boosting, ml_m_boosting, ml_r_boosting)

        self.logger.info("Run XGBoost")
        fit_obj = dml_pliv_obj_boosting.fit()  # Fit and show summary for boosting

        results_df = self.generate_point_inference(fit_obj=fit_obj, model_name="DML: XGBoost", framework="doubleml")

        rmse_xgboost = self.generate_rmse_df(dml_pliv_obj_boosting.evaluate_learners(), "DML: XGBoost")

        return results_df, rmse_xgboost # Display the DataFrame

    def dml_gbm(self, obj_dml_data):
        learner_boosting = GradientBoostingRegressor(n_estimators=20, max_depth=5, learning_rate=0.01, random_state = random.seed(1234), n_jobs=-1)

        ml_l_boosting = clone(learner_boosting)
        ml_m_boosting = clone(learner_boosting)
        ml_r_boosting = clone(learner_boosting)

        # For boosting
        dml_pliv_obj_boosting = dml.DoubleMLPLIV(obj_dml_data, ml_l_boosting, ml_m_boosting, ml_r_boosting)

        self.logger.info("Run GBM")
        #dml_pliv_obj_boosting.fit().summary
        fit_obj = dml_pliv_obj_boosting.fit()  # Fit and show summary for boosting

        results_df = self.generate_point_inference(fit_obj=fit_obj, model_name="DML: Gradient Boosting", framework="doubleml")
                
        return results_df, fit_obj # Display the DataFrame

    def dml_random_forest(self, obj_dml_data):

        # Define a function to create RandomForestRegressor with given parameters
        def create_rf_regressor(params):

            if self.n >= 100:
                params['max_features'] = params['max_features'] + 150
            return RandomForestRegressor(
                n_estimators=params['n_estimators'],
                max_features=params['max_features'],
                max_depth=params['max_depth'],
                min_samples_leaf=params['min_samples_leaf'],
                #random_state=1234,  # Ensure reproducibility
                n_jobs=-1  # Use all available cores
            )
        
        if self.hp_params:
            instrument_key = list(self.hp_params["random_forest"].keys())[2]
            endogenous_key = list(self.hp_params["random_forest"]['ml_l'].keys())[0]

            # Extract hyperparameters for ml_l, ml_m, and ml_r
            ml_l_params = self.hp_params["random_forest"]['ml_l'][endogenous_key][0][0]
            ml_r_params = self.hp_params["random_forest"]['ml_r'][endogenous_key][0][0]
            ml_m_params = self.hp_params["random_forest"][instrument_key][endogenous_key][0][0]  # Adjust this key based on the desired instrument

            # Create the RandomForestRegressor objects
            ml_l = create_rf_regressor(ml_l_params)
            ml_m = create_rf_regressor(ml_m_params)
            ml_r = create_rf_regressor(ml_r_params)

        else:
            learner = RandomForestRegressor(n_estimators=200, max_features=200, max_depth=15, min_samples_leaf=4, random_state = random.seed(1234), n_jobs=-1)
            ml_l = clone(learner)
            ml_m = clone(learner)
            ml_r = clone(learner)

        dml_pliv_obj_rf = dml.DoubleMLPLIV(obj_dml_data, ml_l=ml_l, ml_m=ml_m, ml_r=ml_r)

        self.logger.info("Run Random Forest")

        dml_pliv_obj_rf.fit()

        results_df = self.generate_point_inference(fit_obj=dml_pliv_obj_rf, model_name="DML: Random Forests", framework="doubleml")        
        rmse_rf = self.generate_rmse_df(dml_pliv_obj_rf.evaluate_learners(), "DML: Random Forests")

        return results_df, rmse_rf # Display the DataFrame


    def run_hyperparameter_tuning(self, reading_tuned_hp, simulation_or_empirical):

        def define_tuning(params, model, obj_dml_data):
            logger.info(f"Tuning for model {model}")
            learner = {
                "random_forest":RandomForestRegressor(n_jobs=-1),
                "xgboost":XGBRegressor(n_jobs=-1),
                "lasso": make_pipeline(StandardScaler(), Lasso(max_iter=10000))
            }

            ml_l = clone(learner[model])
            ml_m = clone(learner[model])
            ml_r = clone(learner[model])

            dml_pliv_obj = dml.DoubleMLPLIV(obj_dml_data, ml_l, ml_m, ml_r, n_folds=5)

            # Perform hyperparameter tuning
            dml_pliv_obj.tune(params, search_mode='grid_search')

            with open(REPO_DIR / f"code/hyperparameters/empirical_{model}.json", 'w') as json_file:
                json.dump(dml_pliv_obj.params, json_file, indent=4)

        if reading_tuned_hp == True:
            logger.info("Loading hyperparameters")
            self.hp_params = {}
            
            for model_name in ["random_forest", "xgboost", "lasso"]:
                if simulation_or_empirical == "empirical":
                    path = REPO_DIR / f"code/hyperparameters/empirical_{model_name}.json"
                else:
                    path = REPO_DIR / f"code/hyperparameters/simulation_{model_name}.json"

                with open(path, 'r') as file:
                    # Load the JSON data
                    self.hp_params[model_name]  = json.load(file)

            return True
        
        obj_dml_data = dml.DoubleMLClusterData(
            self.df, 
            y_col=self.y, 
            x_cols=self.x, 
            d_cols=self.d, 
            z_cols=self.z, 
            cluster_cols=self.unit_column
            )

        #################
        ## Random Forest
        #################
        logger.info("Hyperparameter tuning Random Forest")

        par_grids_rf = {
            'ml_l': {
                'n_estimators': [50, 100, 200],
                'max_features': [20, 50, 100],
                'max_depth': [3, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4],
                'ccp_alpha': [0.0, 0.01, 0.1]
            },
            'ml_r': {
                'n_estimators': [50, 100, 200],
                'max_features': [20, 50, 100],
                'max_depth': [3, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4],
                'ccp_alpha': [0.0, 0.01, 0.1]
            },
            'ml_m': {
                'n_estimators': [50, 100, 200],
                'max_features': [20, 50, 100],
                'max_depth': [3, 5, 10, 20],
                'min_samples_leaf': [1, 2, 4],
                'ccp_alpha': [0.0, 0.01, 0.1]
            }
        }
        
        define_tuning(params=par_grids_rf, model="random_forest", obj_dml_data=obj_dml_data)

        # #################
        # ## XGBoost
        # #################
        logger.info("Hyperparameter tuning XGBoost")

        par_grids_xgb = {
            'ml_l': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, 20],
                'learning_rate': [0.01, 0.05, 0.1]
            },
            'ml_m': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, 20],
                'learning_rate': [0.01, 0.05, 0.1]
            },
            'ml_r': {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 10, 20],
                'learning_rate': [0.01, 0.05, 0.1]
            }
        }

        define_tuning(params=par_grids_xgb, model="xgboost", obj_dml_data=obj_dml_data)

        # #################
        # ## LASSO ##
        # #################

        par_grids_lasso = {
            'ml_l': {
                'lasso__alpha': [0.0005, 0.001, 0.01, 0.05, 0.1, 1, 10]
            },
            'ml_m': {
                'lasso__alpha': [0.0005, 0.001, 0.01, 0.05, 0.1, 1, 10]
            },
            'ml_r': {
                'lasso__alpha': [0.0005, 0.001, 0.01, 0.05, 0.1, 1, 10]
            }
        }
        
        define_tuning(params=par_grids_lasso, model="lasso", obj_dml_data=obj_dml_data)

        return "######## Hyperparameter tuning was successful! ########"

    def run_dml_empirical_inference(self, how_many):

        obj_dml_data = dml.DoubleMLClusterData(
            self.df, 
            y_col=self.y, 
            x_cols=self.x, 
            d_cols=self.d, 
            z_cols=self.z, 
            cluster_cols=self.unit_column
            )
        
        list_of_results = []
        merged_rmses = []

        round = 1
        while round <= how_many:

            logger.info(f"#### Simulation: {round}")
            # Run models and store results
            models_to_run = {
                'Lasso': self.dml_lasso,
                'XGBoost': self.dml_xgboost,
                'Random Forest': self.dml_random_forest
            }

            for _, model_function in models_to_run.items():
                results, rmse = model_function(obj_dml_data)

                results["simulation"] = round
                list_of_results.append(results)
                merged_rmses.append(rmse)

            round += 1

        df_results = pd.DataFrame(list_of_results)
        df_rmse = pd.DataFrame(merged_rmses)

        df_results.to_excel(
            REPO_DIR / "data/CONSOLIDATED_results_angrist_dml_noinstitutions.xlsx",
            index=False
        )

        df_rmse.to_excel(
            REPO_DIR / "data/learners_rmse.xlsx",
            index=False
        )