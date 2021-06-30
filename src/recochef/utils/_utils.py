import yaml


def read_yaml(file_path):
  """reads YAML file"""
  with open(file_path, "r") as f:return yaml.safe_load(f)