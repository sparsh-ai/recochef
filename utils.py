import importlib.resources
import yaml

def read_yaml(file_name):
  with importlib.resources.path("recochef.config", file_name) as file_path:
    with open(file_path, "r") as f:
      return yaml.safe_load(f)