from data_cache import pandas_cache
import pandas as pd
import os

from recochef.datasets.dataset import Dataset
from recochef.utils._utils import download_yandex


class Cosmetics(Dataset):
  def __init__(self, version='v1'):
    super(Cosmetics, self).__init__()
    self.version = version

  @pandas_cache
  def load_interactions(self, filepath='cosmetics_interactions.parquet.gzip', nrows=None, chunk=1):
    fileurls = [self.permalinks['cosmetics'][self.version]['chunk1'],
               self.permalinks['cosmetics'][self.version]['chunk2'],
               self.permalinks['cosmetics'][self.version]['chunk3'],
               self.permalinks['cosmetics'][self.version]['chunk4'],
               self.permalinks['cosmetics'][self.version]['chunk5']]
    if not os.path.exists(filepath):
      download_yandex(fileurls[chunk-1], filepath)
    interactions = pd.read_parquet(filepath)
    interactions.columns = ['TIMESTAMP','EVENTTYPE','ITEMID','CATEGORYID',
                            'CATEGORYCODE','BRAND','PRICE','USERID','SESSIONID']
    interactions = interactions[['USERID','ITEMID','EVENTTYPE','TIMESTAMP',
                                 'SESSIONID','CATEGORYID','CATEGORYCODE',
                                 'BRAND','PRICE']]
    return interactions