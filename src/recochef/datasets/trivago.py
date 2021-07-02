from data_cache import pandas_cache
import pandas as pd
import os

from recochef.datasets.dataset import Dataset
from recochef.utils._utils import download_yandex


class Trivago(Dataset):
  def __init__(self, version='v1'):
    super(Trivago, self).__init__()
    self.version = version

  @pandas_cache
  def load_train(self, filepath='trivago_train.parquet.gz', nrows=None):
    fileurl = self.permalinks['trivago'][self.version]['train']
    if not os.path.exists(fileurl):
      download_yandex(fileurl, filepath)
    if nrows:
      train = pd.read_parquet(filepath, nrows=nrows)
    else:
      train = pd.read_parquet(filepath)
    train.columns = ['USERID','SESSIONID','TIMESTAMP','STEP','EVENTTYPE','REFERENCE',
                     'PLATFORM','CITY','DEVICE','FILTERS','IMPRESSIONS','PRICES']
    return train

  @pandas_cache
  def load_test(self, filepath='trivago_test.parquet.gz', nrows=None):
    fileurl = self.permalinks['trivago'][self.version]['test']
    if not os.path.exists(fileurl):
      download_yandex(fileurl, filepath)
    if nrows:
      test = pd.read_parquet(filepath, nrows=nrows)
    else:
      test = pd.read_parquet(filepath)
    test.columns = ['USERID','SESSIONID','TIMESTAMP','STEP','EVENTTYPE','REFERENCE',
                     'PLATFORM','CITY','DEVICE','FILTERS','IMPRESSIONS','PRICES']
    return test

  @pandas_cache
  def load_items(self):
    filepath = self.permalinks['trivago'][self.version]['items']
    items = pd.read_parquet(filepath)
    items.columns = ['ITEMID','PROPERTIES']
    return items

  @pandas_cache
  def load_validation(self):
    filepath = self.permalinks['trivago'][self.version]['validation']
    validation = pd.read_parquet(filepath)
    validation.columns = ['USERID','SESSIONID','TIMESTAMP','STEP','EVENTTYPE','REFERENCE',
                     'PLATFORM','CITY','DEVICE','FILTERS','IMPRESSIONS','PRICES']
    return validation

  @pandas_cache
  def load_confirmation(self):
    filepath = self.permalinks['trivago'][self.version]['confirmation']
    confirmation = pd.read_parquet(filepath)
    confirmation.columns = ['USERID','SESSIONID','TIMESTAMP','STEP','EVENTTYPE','REFERENCE',
                     'PLATFORM','CITY','DEVICE','FILTERS','IMPRESSIONS','PRICES']
    return confirmation