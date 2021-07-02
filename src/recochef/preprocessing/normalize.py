def simple_normalize(data, method='minmax', target_column='RATING'):

  zscore = lambda x: (x - x.mean()) / x.std()
  minmax = lambda x: (x - x.min()) / (x.max() - x.min())

  if method=='minmax':
    norm = data.groupby('USERID')[target_column].transform(minmax)
  elif method=='zscore':
    norm = data.groupby('USERID')[target_column].transform(zscore)
  
  data.loc[:,target_column] = norm
  
  return data