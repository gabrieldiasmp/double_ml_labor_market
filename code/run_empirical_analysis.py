from pathlib import Path
from empirical_analysis.preprocessing_data import ProcessingPipeline
from empirical_analysis.run_models import CausalInferenceModels
from empirical_analysis.report import ReportPipeline

import doubleml as dml

from utils.logging_configs import setup_logging
logger = setup_logging('empirical_analysis.log')

REPO_DIR = Path(__file__).parent.parent

def main():
    
    logger.info("################## STARTING THE PROCESS ##################")

    logger.info("--------- Cleaning data")

    processing_pipeline = ProcessingPipeline()
    df_immigration = processing_pipeline.run()

    # logger.info("--------- Exporting sample to excel")
    # df_immigration.to_excel(REPO_DIR / "data/df_immigration_sample.xlsx", index=False)

    # logger.info("--------- Exploratory analysis")
    # descriptive_table = ReportPipeline(df_immigration, processing_pipeline.features).run()
    # descriptive_table.to_excel(REPO_DIR / "data/descriptive_statistics.xlsx")

    logger.info("--------- Run hyperparameter tuning")
    features_cols = processing_pipeline.features
    models = CausalInferenceModels(
        df=df_immigration, #df_with_diff,
        y_column=features_cols["dependent"][0], 
        d_columns=features_cols["endog"],
        x_columns=features_cols["exog"],
        z_columns=features_cols["instruments"], 
        unit_column='country', 
        time_column=features_cols["year_index_variable"][0],
        desired_alpha=0,
        n=df_immigration['country'].nunique()
        )

    models.run_hyperparameter_tuning()

    logger.info("--------- Run modeling")

    models.run_dml_empirical_inference()

    logger.info("-----  success!! -----")
    return True


if __name__ == "__main__":
    main()

