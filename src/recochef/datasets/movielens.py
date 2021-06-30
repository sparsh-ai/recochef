from data_cache import pandas_cache
import pandas as pd

from recochef.datasets.dataset import Dataset

class MovieLens(Dataset):
  def __init__(self, data='100k', version='v1'):
    super(MovieLens, self).__init__()
    self.data = data
    self.version = version

  @pandas_cache
  def load_interactions(self):
    tag = self.permalinks['movielens'][self.data][self.version]['interactions']
    interactions = pd.read_csv(self.permalinks[tag], low_memory=False)
    interactions.columns = ['USERID','ITEMID','RATING','TIMESTAMP']
    return interactions

  @pandas_cache
  def load_users(self):
    tag = self.permalinks['movielens'][self.data][self.version]['users']
    users = pd.read_csv(self.permalinks[tag], low_memory=False)
    users.columns = ['USERID','AGE','GENDER','OCCUPATION','ZIPCODE']
    return users

  @pandas_cache
  def load_items(self):
    tag = self.permalinks['movielens'][self.data][self.version]['items']
    items = pd.read_csv(self.permalinks[tag], low_memory=False)
    items.columns = ['ITEMID','TITLE','RELEASE','VIDRELEASE','URL',
                            'UNKNOWN', 'ACTION', 'ADVENTURE', 'ANIMATION',
                            'CHILDREN', 'COMEDY', 'CRIME', 'DOCUMENTARY',
                            'DRAMA', 'FANTASY', 'FILMNOIR', 'HORROR', 'MUSICAL',
                            'MYSTERY', 'ROMANCE', 'SCIFI', 'THRILLER', 'WAR',
                            'WESTERN']
    return items