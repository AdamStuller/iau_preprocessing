from sklearn.base import BaseEstimator
import numpy as np

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