import pandas as pd
import numpy as np
from pathlib import Path
from simulation.generate_simulation_dgp import GenerateDGP
from empirical_analysis.run_models import CausalInferenceModels
from utils.import_configs import import_hyperparameters
import argparse

from utils.logging_configs import setup_logging
logger = setup_logging('model_simulation.log')

DATA_DIR = Path(__file__).parent.parent / "data"

def main(args):

    # Simulation settings
    n_values = [50, 100, 200]
    T = 10
    alpha = 0.5
    num_simulations = 100

    logger.info("################## STARTING THE PROCESS ##################")

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
            desired_alpha=alpha, n=n_values
        )

        logger.info("--------- Run hyperparameter tuning")

        models.run_hyperparameter_tuning(
            reading_tuned_hp=args.read_hps,
            simulation_or_empirical="simulation"
        )

    logger.info("-----  success!! -----")
    return True


if __name__ == "__main__":

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Activate hyperparameters")
    
    # Add the "hp_param_activate" argument with a default value of "on"
    parser.add_argument('--read_hps', type=str, default=True, help='Activate hyperparameter mode (default: True)')
    parser.add_argument('--how_many', type=int, default=1, help='How many simulations? (default: 1)')
    parser.add_argument('--interaction_institutions', type=bool, default=False, help='Interaction with institutions?')
    parser.add_argument('--env', type=str, default="dev", help='Dev or prd? If dev, the process will write in tst tables')

    # Parse the command-line arguments
    args = parser.parse_args()

    print("################ ARGS: ", args)

    main(args)

