from data_cache import pandas_cache
import pandas as pd
import pickle
import os

from recochef.datasets.dataset import Dataset
from recochef.utils._utils import download_yandex


class Coveo(Dataset):
  def __init__(self, version='v1'):
    super(Coveo, self).__init__()
    self.version = version

  @pandas_cache
  def load_browsing_events(self, filepath='browsing.parquet.gz'):
    fileurl = self.permalinks['coveo'][self.version]['browsing']
    if not os.path.exists(filepath):
      download_yandex(fileurl, filepath)
    browsing = pd.read_parquet(filepath)
    browsing.columns = ['SESSIONID','EVENTTYPE','ACTIONTYPE','ITEMID','TIMESTAMP','URLID']
    return browsing

  @pandas_cache
  def load_search_events(self, filepath='searching.parquet.gz'):
    fileurl = self.permalinks['coveo'][self.version]['searching']
    if not os.path.exists(filepath):
      download_yandex(fileurl, filepath)
    searching = pd.read_parquet(filepath)
    searching.columns = ['SESSIONID','QUERY_VECTOR','ITEMID_CLICKED','ITEMID_VIEW','TIMESTAMP']
    return searching

  @pandas_cache
  def load_metadata(self, filepath='metadata.parquet.gz'):
    fileurl = self.permalinks['coveo'][self.version]['metadata']
    if not os.path.exists(filepath):
      download_yandex(fileurl, filepath)
    metadata = pd.read_parquet(filepath)
    metadata.columns = ['ITEMID','DESCRIPTION_VECTOR','CATEGORYID','IMAGE_VECTOR','PRICE_BUCKET']
    return metadata

  def load_labels(self, filepath='labels.pickle'):
    fileurl = self.permalinks['coveo'][self.version]['labels']
    if not os.path.exists(filepath):
      download_yandex(fileurl, filepath)
    with open(filepath, 'rb') as f:
      labels = pickle.load(f)
    return labels