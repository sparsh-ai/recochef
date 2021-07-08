from data_cache import pandas_cache
import pandas as pd
import os

from recochef.datasets.dataset import Dataset
from recochef.utils._utils import download_yandex


class WikiRecs(Dataset):
  def __init__(self, version='v1'):
    super(WikiRecs, self).__init__()
    self.version = version

  @pandas_cache
  def load_interactions(self, filepath='wikirecs.parquet.gz', nrows=None):
    fileurl = self.permalinks['wikirecs'][self.version]['interactions']
    if not os.path.exists(filepath):
      download_yandex(fileurl, filepath)
    interactions = pd.read_parquet(filepath)
    interactions.columns = ['USERID','USERNAME','ITEMID','TITLE','TIMESTAMP','SIZEDIFF']
    return interactions