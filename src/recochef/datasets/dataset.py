from data_cache import pandas_cache
import config_with_yaml as config

import warnings
warnings.filterwarnings('ignore')

from recochef.config import permalinks

current_dir = os.path.dirname(os.path.dirname(__file__))
parent_dir = os.path.dirname(current_dir)
parmalink_path = os.path.join(parent_dir, 'config', 'parmalink.yaml')
cfg = config.load(parmalink_path)


class Dataset:
  def __init__(self):
    self.cfg = cfg
  
  @pandas_cache
  def load_interactions(self):
    pass

  @pandas_cache
  def load_users(self):
    pass

  @pandas_cache
  def load_items(self):
    pass