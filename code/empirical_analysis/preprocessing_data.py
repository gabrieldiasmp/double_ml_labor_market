from pathlib import Path

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).parent.parent.parent


class ProcessingPipeline:
    def __init__(self, path = REPO_DIR / "data", interactions_institutions=False):

        self.path = path
        self.interactions_institutions = interactions_institutions

    @staticmethod
    def convert_to_datetime(period):
        year = int(period[:4])
        return pd.to_datetime(f"{year}", format='%Y')
    
    def reading_data(self):

        df_immigration = pd.read_stata(self.path / "df_immigration.dta")
        df_immigration['year_datetime'] = df_immigration['year'][:4]
        # Apply the function to the 'period' column
        df_immigration['year_datetime'] = df_immigration['year'].apply(self.convert_to_datetime)
        df_immigration['year_datetime'] = df_immigration['year_datetime'].dt.year


        df_barent = pd.read_sas(self.path / "bar_ent.sas7bdat")
        df_barent.columns = ["country", "bar_ent"]
        df_barent['country'] = df_barent['country'].apply(lambda x: x.decode('utf-8'))

        df_immigration = pd.merge(df_immigration, df_barent, on='country', how='left')
        
        return df_immigration

    def variable_adjustments(self, df_immigration):
        # df_immigration["emp_prot"]=df_immigration["emp_prot"]-13
        # df_immigration["lab_stan"]=(df_immigration["lab_stan"]-5)/1.9518
        # df_immigration["rep_rate"]=(df_immigration["rep_rate"]-63)/17.70741
        # df_immigration["ep"]=(df_immigration["bar_ent"]-1.715)/.604373

        df_immigration["ep"]=df_immigration["bar_ent"]

        df_immigration['NEU_ep'] = df_immigration['lnf_lf1'] * df_immigration['ep']
        df_immigration['NEU_ls'] = df_immigration['lnf_lf1'] * df_immigration['lab_stan']
        df_immigration['NEU_rr'] = df_immigration['lnf_lf1'] * df_immigration['rep_rate']

        df_immigration['nbos_ep'] = df_immigration['nbospr1'] * df_immigration['ep']
        df_immigration['nowar_ep'] = df_immigration['nowarpr1'] * df_immigration['ep']
        df_immigration['nkos_ep'] = df_immigration['nkospr1'] * df_immigration['ep']

        df_immigration['nbos_ls'] = df_immigration['nbospr1'] * df_immigration['lab_stan']
        df_immigration['nowar_ls'] = df_immigration['nowarpr1'] * df_immigration['lab_stan']
        df_immigration['nkos_ls'] = df_immigration['nkospr1'] * df_immigration['lab_stan']

        df_immigration['nbos_rr'] = df_immigration['nbospr1'] * df_immigration['rep_rate']
        df_immigration['nowar_rr'] = df_immigration['nowarpr1'] * df_immigration['rep_rate']
        df_immigration['nkos_rr'] = df_immigration['nkospr1'] * df_immigration['rep_rate']

        return df_immigration
        
    def defining_model_variables(self, df_immigration):
        # Define the SAS macros as a dictionary
        self.model_features = {
            'non_eu_immigration_share':['lnf_lf1'],
            'years': ['lneu_lf1', 'd84', 'd85', 'd86', 'd87', 'd88', 'd89', 'd90', 'd91', 'd92', 'd93', 'd94', 'd95', 'd96', 'd97', 'd98', 'd99'], #
            'country': ['be', 'dk', 'de91', 'de_91', 'gr', 'es', 'fr', 'ie', 'it', 'lu', 'nl', 'at', 'pt', 'fi', 'se', 'uk', 'no', 'is', 'ch'],
            'ctrends': ['trendbe', 'trendk', 'trende91', 'trend_91', 'trendgr', 'trendes', 'trendfr', 'trendie', 'trendit',
                        'trendlu', 'trendnl', 'trendat', 'trendpt', 'trendfi', 'trendse', 'trenduk', 'trendno', 'trendis',
                        'trendch'],
            'inst1': ['nbospr1', 'nowarpr1', 'nkospr1'],
            "institutions": ['emp_prot','lab_stan','rep_rate'],
            'population_variables': df_immigration.iloc[:, 147:171].columns.tolist(),
            'macroeconomic_variables': df_immigration.iloc[:, 277:280].columns.tolist()+['schengen', 'p96schen']
        }

        self.model_instruments = {
            'first_instruments': ['nbospr1', 'nowarpr1', 'nkospr1'],
            'instls': ['nbos_ls', 'nowar_ls', 'nkos_ls'],
            'instrr': ['nbos_rr', 'nowar_rr', 'nkos_rr'],
            'instlsrr': ['nbos_ls', 'nowar_ls', 'nkos_ls', 'nbos_rr', 'nowar_rr', 'nkos_rr'],
            'inst1b': ['nbosds12', 'noward12', 'nkosds12', 'nbos_ep', 'nowar_ep', 'nkos_ep'],
            'inst2b': ['nbosds12', 'noward12', 'nkosds12', 'nbos_ep', 'nowar_ep', 'nkos_ep', 'nbos_ls', 'nowar_ls', 'nkos_ls'],
            'inst3b': ['nbosds12', 'noward12', 'nkosds12', 'nbos_ep', 'nowar_ep', 'nkos_ep','nbos_ls', 'nowar_ls', 'nkos_ls', 'nbos_rr', 'nowar_rr', 'nkos_rr'] # 
        }

        # Define endog, exog, and instruments based on the interactions_institutions flag
        if self.interactions_institutions:
            endog = self.model_features["non_eu_immigration_share"] + ['NEU_ep', 'NEU_ls', 'NEU_rr']
            exog = (self.model_features["years"] +
                    self.model_features["ctrends"] +
                    self.model_features["country"]+
                    self.model_features["population_variables"])
            instruments = self.model_instruments["inst3b"]
        else:
            endog = self.model_features["non_eu_immigration_share"]
            exog = (self.model_features["country"] + 
                    self.model_features["years"] + 
                    self.model_features["ctrends"] + 
                    self.model_features["population_variables"] + 
                    self.model_features["macroeconomic_variables"])
            instruments = self.model_instruments["first_instruments"]
            
        self.features = {
            "dependent": ['lne_p'],
            "endog": endog,
            "exog": exog,
            "instruments": instruments,
            "year_index_variable": ['year_datetime'],
            "institutions": self.model_features["institutions"]
        }

    def get_sample(self, df_immigration):

        # Filter the data
        data_filtered = df_immigration[
            (df_immigration['is'] == 0) & \
            (df_immigration['dman'] == 1) & \
            (df_immigration['dold'] == 0)]

        return data_filtered

    def filter_needed_columns_for_inference(self, data_filtered):

        list_of_variables_to_be_selected = {
            "with_institutions": (
                ["country"]+
                self.features["dependent"]+
                self.features["endog"]+
                self.model_features["institutions"]+
                self.features["exog"]+
                self.features["instruments"]+
                self.features["year_index_variable"]),
            "without_institutions": (
                ["country"]+
                self.features["dependent"]+
                self.features["endog"]+
                self.features["exog"]+
                self.features["instruments"]+
                self.features["year_index_variable"])
        }

        if self.interactions_institutions == 'True':
            with_institutions_or_not = "with_institutions"
        else:
            with_institutions_or_not = "without_institutions"

        data_filtered = data_filtered[list_of_variables_to_be_selected[with_institutions_or_not]].dropna()
        #country_without_is = [i for i in self.model_features["country"] if i not in ["is", 'ch', 'lu', 'gr']]

        return data_filtered

    def run(self):

       df_immigration = self.reading_data()

       self.defining_model_variables(df_immigration)

       df_immigration = self.variable_adjustments(df_immigration)

       df_immigration_sample = self.get_sample(df_immigration)

       return df_immigration_sample


