from data_cache import pandas_cache
import warnings
import os

from recochef.utils._utils import read_yaml

warnings.filterwarnings('ignore')

parent_dir = os.path.dirname(os.path.dirname(__file__))
parmalink_path = os.path.join(parent_dir, 'config', 'parmalink.yaml')
permalinks = read_yaml(parmalink_path)

class Dataset:
  def __init__(self):
    self.permalinks = permalinks
  
  @pandas_cache
  def load_interactions(self):
    pass

  @pandas_cache
  def load_users(self):
    pass

  @pandas_cache
  def load_items(self):
    pass