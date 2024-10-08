from pathlib import Path
import argparse
from empirical_analysis.preprocessing_data import ProcessingPipeline
from empirical_analysis.run_models import CausalInferenceModels
from empirical_analysis.report import ReportPipeline

import doubleml as dml

from utils.logging_configs import setup_logging
logger = setup_logging('empirical_analysis.log')

REPO_DIR = Path(__file__).parent.parent

def main(args):
    
    logger.info("################## STARTING THE PROCESS ##################")

    logger.info("--------- Cleaning data")

    processing_pipeline = ProcessingPipeline(
        interactions_institutions=args.interaction_institutions)
    df_immigration = processing_pipeline.run()
    print("###############SHAPE AFTER PREPROCESSING: ", df_immigration.shape)

    # logger.info("--------- Exporting sample to excel")
    # df_immigration.to_excel(REPO_DIR / "data/df_immigration_sample.xlsx", index=False)

    # logger.info("--------- Exploratory analysis")
    # descriptive_table = ReportPipeline(df_immigration, processing_pipeline.features).run()
    # descriptive_table.to_excel(REPO_DIR / "data/tst_descriptive_statistics.xlsx")

    logger.info("--------- Run modeling")
    features_cols = processing_pipeline.features

    df_filtered = processing_pipeline.filter_needed_columns_for_inference(df_immigration)
    print("###############SHAPE AFTER PREPROCESSING: ", df_filtered.shape)

    models = CausalInferenceModels(
        df=df_filtered,
        y_column=features_cols["dependent"][0], 
        d_columns=features_cols["endog"],
        x_columns=features_cols["exog"],
        z_columns=features_cols["instruments"], 
        unit_column='country', 
        time_column=features_cols["year_index_variable"][0],
        desired_alpha=0,
        n=df_filtered['country'].nunique(),
        env=args.env
        )

    logger.info("--------- Run hyperparameter tuning")

    models.run_hyperparameter_tuning(
        reading_tuned_hp=args.read_hps,
        simulation_or_empirical="empirical"
    )

    logger.info("--------- Run inference")

    models.run_dml_empirical_inference(
        how_many=args.how_many,
        with_institutions=args.interaction_institutions)

    logger.info("-----  success!! -----")
    return True


if __name__ == "__main__":

    # Create the argument parser
    parser = argparse.ArgumentParser(description="Activate hyperparameters")
    
    # Add the "hp_param_activate" argument with a default value of "on"
    parser.add_argument('--read_hps', type=str, default='True', help='Activate hyperparameter mode (default: True)')
    parser.add_argument('--how_many', type=int, default=1, help='How many simulations? (default: 1)')
    parser.add_argument('--interaction_institutions', type=str, default='False', help='Interaction with institutions?')
    parser.add_argument('--env', type=str, default="dev", help='Dev or prd? If dev, the process will write in tst tables')

    # Parse the command-line arguments
    args = parser.parse_args()

    main(args)

