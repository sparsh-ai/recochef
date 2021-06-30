from data_cache import pandas_cache
import pandas as pd

from recochef.datasets.dataset import Dataset

class BookCrossing(Dataset):
  def __init__(self, version='v1'):
    super(BookCrossing, self).__init__()
    self.version = version

  @pandas_cache
  def load_interactions(self):
    tag = self.cfg.getProperty('bookcrossing.{}.interactions'.format(self.version))
    interactions = pd.read_csv(self.permalinks[tag],
                               sep=';',
                               error_bad_lines=False,
                               warn_bad_lines=False,
                               encoding='latin-1',
                               low_memory=False)
    interactions.columns = ['USERID','ITEMID','RATING']
    return interactions

  @pandas_cache
  def load_users(self):
    tag = self.cfg.getProperty('bookcrossing.{}.users'.format(self.version))
    users = pd.read_csv(self.permalinks[tag],
                        sep=';',
                        error_bad_lines=False,
                        warn_bad_lines=False,
                        encoding='latin-1',
                        low_memory=False)
    users.columns = ['USERID', 'LOCATION', 'AGE']
    return users

  @pandas_cache
  def load_items(self):
    tag = self.cfg.getProperty('bookcrossing.{}.items'.format(self.version))
    items = pd.read_csv(self.permalinks[tag],
                        sep=';',
                        error_bad_lines=False,
                        warn_bad_lines=False,
                        encoding='latin-1',
                        low_memory=False)
    items.columns = ['ITEMID', 'TITLE', 'AUTHOR', 'YEAR', 'PUBLISHER',
                     'URLSMALL', 'URLMEDIUM', 'URLLARGE']
    return items