def simple_normalize(data, method='minmax'):

  zscore = lambda x: (x - x.mean()) / x.std()
  minmax = lambda x: (x - x.min()) / (x.max() - x.min())

  if method=='minmax':
    norm = data.groupby('USERID').RATING.transform(minmax)
  elif method=='zscore':
    norm = data.groupby('USERID').RATING.transform(zscore)
  
  data.loc[:,'RATING'] = norm
  
  return data