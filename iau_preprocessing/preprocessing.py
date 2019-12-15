from sklearn.base import BaseEstimator
from functools import reduce
import pandas as pd
import numpy as np
import re
import math

class NaValuesImputer(BaseEstimator):
  def __init__(self, Clf):
    self.Clf = Clf
    self.classifiers = {}

  def __has_na(self, x):
    uniq = x.isna().unique()
    return len(uniq) != 1

  def __fill_column(self, df, column):
    if self.__has_na(df[column]):
      copied_df = df.copy(deep=True)
      
      na_rows, not_na_rows = copied_df[copied_df[column].isna()], copied_df[copied_df[column].notna()]
      na_rows[column] = self.classifiers[column].predict( # now we predict missing values
          na_rows                       # with na rows as x
            .drop(columns=[column])     # dropped of column
            .fillna(na_rows.median())   # and with na values filled with median
            )

      return pd.concat([not_na_rows, na_rows]).sort_index()
    else:
      return df

  def __fit_column(self, df, column):
    copied_df = df.copy(deep=True)
    not_na_rows = copied_df[copied_df[column].notna()]
    return self.Clf().fit(                  # fits passed classifier, each time new one
        not_na_rows                         # x are all not na rows
          .drop(columns=[column])           # dropped of column to be imputed
          .fillna(not_na_rows.median()),    # and all missing walues replaced by median
        not_na_rows[column]                 # y are not na rows of column
    )

  def fit(self, x, y=None):
    for column in [ *x.columns ]:
      self.classifiers[column] = self.__fit_column(x, column)
    return self

  def transform(self, x):
    return reduce(lambda df, column: self.__fill_column(df, column), [*x.columns], x)

class OneHotTransformer(BaseEstimator):
  def __init__(self, columns):
    self.columns = columns    # Categorical 
    self.all_columns = []     # All columns of trainning data, gets filled in fit method
    self.categories = []      # This gets filled while fitting, contains all caegories created by get_dummies on training data

  def __update_categories(self, df):
    more_columns=filter(lambda x: x not in self.all_columns and x not in self.categories, [*df.columns]) 
    df=df.drop(columns=more_columns)                                                                     

    for missing_row in filter(lambda x: x not in [*df.columns], self.categories):                       
      df[missing_row] = 0                                                                                
    return df

  def __column_one_hot(self, df, column):
      encoded = pd.get_dummies(df[column])
      return pd.merge(df, encoded, left_index=True, right_index=True).drop(columns=column).rename(columns=lambda x: x.lower())

  def __get_categories(self, df, column):
      return [*pd.get_dummies(df[column]).columns]

  def fit(self, x, y=None):
    self.all_columns=[*x.columns]
    for column in self.columns:
      for cat in self.__get_categories(x, column):
        self.categories.append(cat)
    print(self.categories)
    return self

  def transform(self, x):
    return self.__update_categories(
        reduce(lambda df, column: self.__column_one_hot(df, column), self.columns, x)
        )

class OutlierReplacer():
  def __init__(self):
    self.columns_to_normalize = ['skewness_glucose', 'kurtosis_glucose', 'skewness_oxygen', 'kurtosis_oxygen','mean_glucose', 'mean_oxygen','std_oxygen', 'std_glucose'] # columns to be normalized by sqrt 

  def __sqrt_columns(self,df):                  # normalizing values with sqrt()
    for column in self.columns_to_normalize:
      df[column] =  np.sqrt(df[column])
    return df

  def __replace_with_quantile(self,df):         # tbd
    for column in self.columns_to_normalize:
      whisker_l, whisker_r = self.__get_column_whiskers(df, column)
      if sum(df[column] < whisker_l) < len(df)*0.01:
        for index, row in df.iterrows():
          if row[column] < whisker_l:
            df.loc[index,column] = df[column].quantile(0.05)
      if sum(df[column] > whisker_r) < len(df)*0.01:
        for index, row in df.iterrows():
          if row[column] > whisker_r:
            df.loc[index,column] = df[column].quantile(0.95)
    return df
    
  def __get_column_whiskers(self, df, column):
    descr = df[column].describe()
    whisker_r = np.min([descr['max'], descr['75%'] + (1.5 * (descr['75%'] - descr['25%']))])
    whisker_l = np.max([descr['min'], descr['25%'] - (1.5 * (descr['75%'] - descr['25%']))])
    return whisker_l, whisker_r

  def __replace_outliers(self, df):
       
      return self.__replace_with_quantile(self.__sqrt_columns(df))

  def transform(self,df):
    return self.__replace_outliers(df)

  def fit(self, x, y=None):
    return self

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

class PregnancyTransformer(BaseEstimator):
  def __init__(self, columns=['pregnant']):
    self.columns = columns if len(columns)>1 else columns[0]

  def __normalize_pregnancy(self, x):
    if not isinstance(x, str):
        return x
    x = re.sub(r"^f$|(^F$)|(^FALSE$)", r"0", x)
    x = re.sub(r"^t$|^T$|^TRUE$", r"1", x)
    return int(x)

  def fit(self, x, y=None):
    return self

  def transform(self, x):
    copied_df = x.copy(deep=True)
    copied_df.pregnant = copied_df.pregnant.apply(self.__normalize_pregnancy)
    return copied_df

class ValueNormalizer(BaseEstimator):

  def __init__(self):
    self.columns_to_be_normalized=['skewness_glucose', 'kurtosis_glucose', 'skewness_oxygen', 'kurtosis_oxygen'] 
    self.columns_abs = ['mean_glucose', 'mean_oxygen','std_oxygen', 'std_glucose']
    self.columns_to_be_repaired = ['std_oxygen', 'std_glucose']

  def __normalize_std(self,df):               # normalizing standart deviation, because std can't be less than 0 
    for index, row in df.iterrows():
      for column in self.columns_to_be_repaired:
        df.loc[index,column] = abs(row[column])
        if row[column] >1000:                             # some std values are too high. It is probably because of mistake done when inserting data
          df.loc[index,column] = row[column]/100          # so we are correcting that mistake
    return df

  def __abs_columns(self,df):                             # these columns cant be lower than 0, so we are fixing that mistake
    for column in self.columns_abs:
      df[column] = abs(df[column])
    return df

  def __shift_values(self,df):                            # we are shifting values of this columns so they can be normalized later 
    for column in self.columns_to_be_normalized:
      df[column] = df[column] - math.floor(min(df[column]))
    return df

  def __normalize_values(self,df):
    return self.__normalize_std(self.__abs_columns(self.__shift_values(df))) 
     

  def transform(self,df):
    return self.__normalize_values(df)

  def fit(self, x, y=None):
    return self
