from sklearn.base import BaseEstimator
from functools import reduce
import pandas as pd

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