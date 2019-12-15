from sklearn.base import BaseEstimator
import math

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
