from data_cache import pandas_cache
import pandas as pd
import os

from recochef.datasets.dataset import Dataset
from recochef.utils._utils import download_yandex


class Tmall(Dataset):
  def __init__(self, version='v1'):
    super(Tmall, self).__init__()
    self.version = version

  @pandas_cache
  def load_interactions(self, filepath='tmall_interactions.csv'):
    fileurl = self.permalinks['tmall'][self.version]['interactions']
    if not os.path.exists(fpath):
      download_yandex(fileurl, filepath)
    interactions = pd.read_csv(filepath,
                               sep=';',
                               error_bad_lines=False,
                               warn_bad_lines=False,
                               low_memory=False)
    # interactions.columns = ['USERID','ITEMID','RATING']
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
    items = pd.read_csv(filepath,
                        sep=';',
                        error_bad_lines=False,
                        warn_bad_lines=False,
                        encoding='latin-1',
                        low_memory=False)
    items.columns = ['ITEMID', 'TITLE', 'AUTHOR', 'YEAR', 'PUBLISHER',
                     'URLSMALL', 'URLMEDIUM', 'URLLARGE']
    return items