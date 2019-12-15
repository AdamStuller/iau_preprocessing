from sklearn.base import BaseEstimator
import re
import pandas as pd
import numpy as np

class PersonalInfoParser(BaseEstimator):
    def __parse_personal_info(self, x):
        if isinstance(x, str):
            return map(lambda x: np.nan if x.strip() == '?' else x.strip(), re.sub(r"(\r(\r)?\n)|(--)", "|", x).split("|"))
        return [np.nan for x in range(0, 5)]

    def fit(self, x, y=None):
        return self

    def transform(self, x):
        personal_info = {
            'occupation':[],
            'country':[],
            'relationship_status':[],
            'race':[],
            'employment': []
        }
        for occupation, country, state, employment,  rase in x['personal_info'].apply(self.__parse_personal_info):
            personal_info['occupation'].append(occupation)
            personal_info['country'].append(country)
            personal_info['relationship_status'].append(state)
            personal_info['race'].append(rase)
            personal_info['employment'].append(employment)

        personal_info_frame = pd.DataFrame(personal_info)
        x.index = personal_info_frame.index

        return pd.merge(x.reindex(), personal_info_frame, right_index=True, left_index=True).drop(columns='personal_info')  