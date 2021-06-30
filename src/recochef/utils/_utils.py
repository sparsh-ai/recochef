import importlib.resources
import requests
import yaml

def read_yaml(file_name):
  with importlib.resources.path("recochef.config", file_name) as file_path:
    with open(file_path, "r") as f:
      return yaml.safe_load(f)

def download_yandex(sharing_link, file_path):
  API_ENDPOINT = 'https://cloud-api.yandex.net/v1/disk/public/resources/download?public_key={}'
  pk_request = requests.get(API_ENDPOINT.format(sharing_link))
  r = requests.get(pk_request.json()['href'])
  open(file_path, 'wb').write(r.content)