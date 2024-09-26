import pandas as pd
import numpy as np
from pathlib import Path
from run_simulations import GenerateDGP
from run_models import CausalInferenceModels
from utils.import_configs import import_hyperparameters

from logging_configs import setup_logging_simulation
logger = setup_logging_simulation('model_simulation.log')

DATA_DIR = Path(__file__).parent.parent / "data"

def main():
    # Simulation settings
    n_values = [15, 50, 100, 200]
    T = 10
    alpha = 0.5
    num_simulations = 30

    hp_parameters = import_hyperparameters()

    list_of_results = []

    for sim in range(num_simulations):
        logger.info(f"############# Simulation {sim + 1}/{num_simulations} #############")

        for n in n_values:
            logger.info(f"------ Number of units: {n} -----------")
            
            # Instantiate DGP class
            generate_dgp = GenerateDGP(n=n, T=T, alpha=alpha)

            # Specify number of controls and instruments
            num_controls = int((n * T) * 0.5)
            num_instruments = 10

            # Generate data
            df = generate_dgp.generate_post_selection_regularization_dgp(n=n, px=num_controls, pz=num_instruments)

            # Extract x_columns
            x_columns = [column for column in df.columns if column.split("_")[0] == "x"]

            # Instantiate model's class
            models = CausalInferenceModels(
                df=df, y_column='y', d_columns='d', x_columns=x_columns, unit_column='unit', time_column='t', 
                desired_alpha=alpha, n=n, hp_params=hp_parameters
            )

            # DML setup
            dml_data = models.prepare_dml_data()

            # Run models and store results
            models_to_run = {
                'Pooled OLS': models.pooled_2sls,
                'FE 2SLS': models.first_difference_2sls,
                'Lasso': models.dml_lasso,
                'XGBoost': models.dml_xgboost,
                'Random Forest': models.dml_random_forest
            }

            for model_name, model_function in models_to_run.items():
                if model_name in ['Pooled OLS', 'FE 2SLS']:
                    results = model_function()
                else:
                    results = model_function(dml_data)

                results["simulation"] = sim + 1
                list_of_results.append(results)

    # Convert the list of results into a DataFrame
    df_results = pd.DataFrame(list_of_results)

    # Define the path to save the Excel file
    path = DATA_DIR / "results_simulation_controls_and_instruments.xlsx"
    df_results.to_excel(path, index=False)
    
    logger.info("----- SUCCESS!! ----")

    return "success!!!"

if __name__ == "__main__":
    main()
