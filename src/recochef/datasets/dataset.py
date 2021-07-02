from data_cache import pandas_cache
import warnings

from recochef.utils._utils import read_yaml

warnings.filterwarnings('ignore')
permalinks = read_yaml('permalink.yaml')


class Dataset:
  def __init__(self):
    self.permalinks = permalinks