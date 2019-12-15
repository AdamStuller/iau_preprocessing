import pandas as pd

def join_datasets(other, personal):  #akoze tento hnus funguje najlepse zatial        
  merged_dataset = pd.merge(other, personal, on='name')
  duplicates=merged_dataset[merged_dataset['name'].duplicated(keep=False)]
  i = -1
  for index, row in duplicates.iterrows():
    i = i+1
    for index2, row2 in duplicates.iterrows():
      if index == index2-1 and i%2 == 0:
        for column in duplicates.columns:
          if pd.isna(row[column]):
            duplicates.loc[index,column] = row2[column]
      if i%2 == 1: 
        duplicates.loc[index,'name'] = 'marked'
  duplicates = duplicates[duplicates.name != 'marked']
  dataset = pd.concat([duplicates,merged_dataset])
  return dataset.drop_duplicates(['name', 'address_x']).sort_index().rename(columns={'address_x': 'address'}).drop(columns='address_y')

def drop_pointless(df, columns):
  return df.drop(columns=columns)

def drop_na_class(df):
  return df[(~ df['class'].isna())].reindex()