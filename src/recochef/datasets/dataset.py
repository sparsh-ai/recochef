from data_cache import pandas_cache
import config_with_yaml as config
import os

import warnings
warnings.filterwarnings('ignore')

parent_dir = os.path.dirname(os.path.dirname(__file__))
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