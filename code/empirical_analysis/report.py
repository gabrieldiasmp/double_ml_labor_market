from pathlib import Path

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).parent.parent.parent

class ReportPipeline:
    def __init__(self, df_immigration, features, path = REPO_DIR / "data"):
        self.path = path
        self.df_immigration = df_immigration
        self.features = features

    def generate_descriptive_statistics(self):
        df_immigration_statistics = self.df_immigration.copy()
        df_immigration_statistics = df_immigration_statistics.loc[
            :, 
            ["country"]+self.features["dependent"]+self.features["endog"]+self.features["exog"]+self.features["institutions"]]

        df_immigration_statistics["non_eu_immigrant_share"] = np.exp(df_immigration_statistics[self.features["endog"][0]])
        df_immigration_statistics["eu_immigrant_share"] = np.exp(df_immigration_statistics["lneu_lf1"])
        df_immigration_statistics["employment"] = np.exp(df_immigration_statistics[self.features["dependent"]])


        df_immigration_statistics = df_immigration_statistics.loc[:, ["country", "employment", "non_eu_immigrant_share", "eu_immigrant_share", "tpop1",'emp_prot', 'lab_stan', 'rep_rate']]

        statistics_table = df_immigration_statistics.\
            groupby('country').describe().\
            loc[:, (slice(None), ['count', 'mean', 'std'])]

        # Split variables into two tables
        table_1_variables = ["employment", "non_eu_immigrant_share", "eu_immigrant_share", "tpop1"]
        table_2_variables = ['emp_prot', 'lab_stan', 'rep_rate']

        # Table 1 with count, mean, std for first 3 variables
        desc_stats_table_1 = statistics_table[table_1_variables].loc[:, (slice(None), ['count', 'mean', 'std'])]

        # Table 2 with count, mean, std for the next 3 variables
        desc_stats_table_2 = statistics_table[table_2_variables].loc[:, (slice(None), ['count', 'mean', 'std'])]

        # Merge the two tables by rows
        merged_stats = pd.concat([desc_stats_table_1, desc_stats_table_2], axis=1)

        print(merged_stats)
        return merged_stats
    
    def run(self):
        return self.generate_descriptive_statistics()

