from sklearn.base import BaseEstimator
import re

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
