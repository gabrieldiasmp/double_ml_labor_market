from pathlib import Path

REPO_DIR = Path(__file__).parent.parent.parent

class ReportPipeline:
    def __init__(self, df_immigration, features, path = REPO_DIR / "data"):
        self.path = path
        self.df_immigration = df_immigration
        self.features = features

    def generate_descriptive_statistics(self):
        statistics_table = self.df_immigration.loc[:, self.features["dependent"]+self.features["endog"]]
        return statistics_table
    
    def run(self):
        return self.generate_descriptive_statistics()

