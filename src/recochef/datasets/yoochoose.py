from data_cache import pandas_cache
import pandas as pd
import os

from recochef.datasets.dataset import Dataset
from recochef.utils._utils import download_yandex


class YooChoose(Dataset):
  def __init__(self, version='v1'):
    super(YooChoose, self).__init__()
    self.version = version

  @pandas_cache
  def load_clicks(self, filepath='yoochoose_clicks.zip', nrows=None):
    fileurl = self.permalinks['yoochoose'][self.version]['clicks']
    if not os.path.exists(filepath):
      download_yandex(fileurl, filepath)
    if nrows:
      clicks = pd.read_csv(filepath,
                           nrows=nrows,
                           header=None)
    else:
      clicks = pd.read_csv(filepath,
                           header=None)
    clicks.columns = ['SESSIONID', 'TIMESTAMP', 'ITEMID', 'CATEGORYID']
    return clicks

  @pandas_cache
  def load_buys(self):
    filepath = self.permalinks['yoochoose'][self.version]['buys']
    buys = pd.read_csv(filepath,
                       header=None)
    buys.columns = ['SESSIONID', 'TIMESTAMP', 'ITEMID', 'PRICE', 'QUANTITY']
    return buys