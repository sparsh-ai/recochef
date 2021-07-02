import numpy as np
import pandas as pd


def simple_aggregate(data,
                     drop_duplicates=True,
                     by='AFFINITY',
                     aggregation='mean',
                     weights=None,
                     half_life=None,
                     ):
  
  if drop_duplicates:
    data = data.drop_duplicates()
  
  if by in data.columns:

    if weights:
      data = data.replace({by:weights})

    if half_life:
      t_ref = pd.to_datetime(data['TIMESTAMP']).max()
      data[by] = data.apply(lambda x: x[by] * np.power(0.5, (t_ref - pd.to_datetime(x['TIMESTAMP'])).days / half_life), axis=1)

    data = data.groupby(['USERID', 'ITEMID']).agg({by: aggregation}).reset_index()

  else:

    data[by] = 1
    data = data.groupby(['USERID', 'ITEMID']).agg({by: 'count'}).reset_index()

  return data