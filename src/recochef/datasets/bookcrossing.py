from data_cache import pandas_cache
import pandas as pd

from recochef.datasets.dataset import Dataset


class BookCrossing(Dataset):
  def __init__(self, version='v1'):
    super(BookCrossing, self).__init__()
    self.version = version

  @pandas_cache
  def load_interactions(self):
    filepath = self.permalinks['bookcrossing'][self.version]['interactions']
    interactions = pd.read_csv(filepath,
                               sep=';',
                               error_bad_lines=False,
                               warn_bad_lines=False,
                               encoding='latin-1',
                               low_memory=False)
    interactions.columns = ['USERID','ITEMID','RATING']
    return interactions

  @pandas_cache
  def load_users(self):
    filepath = self.permalinks['bookcrossing'][self.version]['users']
    users = pd.read_csv(filepath,
                        sep=';',
                        error_bad_lines=False,
                        warn_bad_lines=False,
                        encoding='latin-1',
                        low_memory=False)
    users.columns = ['USERID', 'LOCATION', 'AGE']
    return users

  @pandas_cache
  def load_items(self):
    filepath = self.permalinks['bookcrossing'][self.version]['items']
    items = pd.read_csv(items,
                        sep=';',
                        error_bad_lines=False,
                        warn_bad_lines=False,
                        encoding='latin-1',
                        low_memory=False)
    items.columns = ['ITEMID', 'TITLE', 'AUTHOR', 'YEAR', 'PUBLISHER',
                     'URLSMALL', 'URLMEDIUM', 'URLLARGE']
    return items