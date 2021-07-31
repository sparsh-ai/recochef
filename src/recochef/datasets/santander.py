from data_cache import pandas_cache
import pandas as pd
import pickle
import os

from recochef.datasets.dataset import Dataset
from recochef.utils._utils import download_yandex


class Santander(Dataset):
  def __init__(self, version='v1'):
    super(Santander, self).__init__()
    self.version = version

  @pandas_cache
  def load_train(self, filepath='train.parquet.gz'):
    fileurl = self.permalinks['santander'][self.version]['train']
    if not os.path.exists(filepath):
      download_yandex(fileurl, filepath)
    train = pd.read_parquet(filepath)
    return train

  @pandas_cache
  def load_test(self, filepath='test.parquet.gz'):
    fileurl = self.permalinks['santander'][self.version]['test']
    if not os.path.exists(filepath):
      download_yandex(fileurl, filepath)
    test = pd.read_parquet(filepath)
    return test

  @pandas_cache
  def load_submission(self, filepath='submission.parquet.gz'):
    fileurl = self.permalinks['santander'][self.version]['submission']
    if not os.path.exists(filepath):
      download_yandex(fileurl, filepath)
    submission = pd.read_parquet(filepath)
    return submission