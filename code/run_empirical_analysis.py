from pathlib import Path
from empirical_analysis.preprocessing_data import ProcessingPipeline
from empirical_analysis.run_models import CausalInferenceModels
from empirical_analysis.report import ReportPipeline

from utils.logging_configs import setup_logging
logger = setup_logging('empirical_analysis.log')

REPO_DIR = Path(__file__).parent.parent

def main():
    
    
    ### 
    logger.info("--------- Cleaning data")

    processing_pipeline = ProcessingPipeline()
    df_immigration = processing_pipeline.run()

    logger.info("--------- Exporting sample to excel")
    df_immigration.to_excel(REPO_DIR / "data/df_immigration_sample.xlsx", index=False)

    # logger.info("--------- Exploratory analysis")
    # exploratory_analysis = ReportPipeline(df_immigration, processing_pipeline.features).run()


    ### Run modeling


    return df_immigration


if __name__ == "__main__":
    
    print(main())

