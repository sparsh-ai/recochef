import os
import requests 

def create_data_dir(path):
    if not os.path.isdir(path):
        os.makedirs(path)
        

def download(url, dest_path):
    req = requests.get(url, stream=True)
    req.raise_for_status()
    with open(dest_path, "wb") as fd:
        for chunk in req.iter_content(chunk_size=2 ** 20):
            fd.write(chunk)