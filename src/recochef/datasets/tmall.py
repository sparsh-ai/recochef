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
  def load_interactions(self, filepath='tmall_interactions.zip', nrows=None):
    fileurl = self.permalinks['tmall'][self.version]['interactions']
    if not os.path.exists(fileurl):
      download_yandex(fileurl, filepath)
    if nrows:
      interactions = pd.read_csv(filepath, nrows=nrows)
    else:
      interactions = pd.read_csv(filepath)
    interactions.columns = ['USERID', 'MERCHANTID', 'ITEMID', 'CATEGORYID',
                            'EVENTTYPE', 'TIMESTAMP']
    return interactions

  @pandas_cache
  def load_train(self):
    filepath = self.permalinks['tmall'][self.version]['train']
    train = pd.read_csv(filepath)
    train.columns = ['USERID', 'MERCHANTID', 'LOCATION', 'TIMESTAMP']
    return train

  @pandas_cache
  def load_test(self):
    filepath = self.permalinks['tmall'][self.version]['test']
    test = pd.read_csv(filepath)
    test.columns = ['USERID', 'LOCATION', 'MERCHANTIDS']
    return test

  @pandas_cache
  def load_merchants(self):
    filepath = self.permalinks['tmall'][self.version]['merchants']
    merchants = pd.read_csv(filepath)
    merchants.columns = ['MERCHANTID', 'BUDGET', 'LOCATIONS']
    return merchants