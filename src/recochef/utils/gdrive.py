from pydrive.auth import GoogleAuth
from pydrive.drive import GoogleDrive
from google.colab import auth
from oauth2client.client import GoogleCredentials

import os
import subprocess
from pathlib import Path

__all__ = ['GoogleDriveHandler']

class GoogleDriveHandler:
    """ref - https://gist.github.com/yt114/dc5d2fd4437f858bb73e38f0aba362c7"""
    def __init__(self):
        auth.authenticate_user()
        gauth = GoogleAuth()
        gauth.credentials = GoogleCredentials.get_application_default()
        self.drive = GoogleDrive(gauth)

    def path_to_id(self, rel_path, parent_folder_id='root'):
        rel_path = '/'.join(list(filter(len, rel_path.split('/'))))
        if rel_path == '':
            return parent_folder_id
        else:
            first, *rest = list(filter(len, rel_path.split('/')))
            file_dict = {f['title']:f for f in self.list_folder(parent_folder_id)}
            if first not in file_dict:
                raise Exception('{0} not exist'.format(first))
            else:
                return self.path_to_id('/'.join(rest), file_dict[first]['id'])
            
    def list_folder(self, root_folder_id='root', max_depth=0):
        query = "'{0}' in parents and trashed=false".format(root_folder_id)
        file_list, folder_type = [], 'application/vnd.google-apps.folder'
        for f in self.drive.ListFile({'q': query}).GetList():
            if f['mimeType'] == folder_type and max_depth > 0:
                file_list.append(
                    {
                        'title': f['title'], 
                        'id': f['id'], 
                        'link': f['alternateLink'], 
                        'mimeType': f['mimeType'],
                        'children': self.list_folder(f['id'], max_depth-1)
                    }
                )
            else:
                file_list.append(
                    {
                        'title':f['title'], 
                        'id': f['id'], 
                        'link':f['alternateLink'],
                        'mimeType': f['mimeType']
                    }
                )
        return file_list

    def create_folder(self, folder_name, parent_path=''):
        parent_folder_id = self.path_to_id(parent_path)
        folder_type = 'application/vnd.google-apps.folder'
        file_dict = {f['title']:f for f in self.list_folder(parent_folder_id)}
        if folder_name not in file_dict:
            folder_metadata = {
                'title' : folder_name, 
                'mimeType' : folder_type,
                'parents': [{'kind': 'drive#fileLink', 'id': parent_folder_id}]
            }
            folder = self.drive.CreateFile(folder_metadata)
            folder.Upload()
            return folder['id']
        else:
            if file_dict[folder_name]['mimeType'] != folder_type:
                raise Exception('{0} already exists as a file'.format(folder_name))
            else:
                print('{0} already exists'.format(folder_name))
            return file_dict[folder_name]['id']

    def upload(self, local_file_path, parent_path='', overwrite=True):
        parent_folder_id = self.path_to_id(parent_path)
        file_dict = {f['title']:f for f in self.list_folder(parent_folder_id)}
        file_name = local_file_path.split('/')[-1]
        if file_name in file_dict and overwrite:
            file_dict[file_name].Delete()
        file = self.drive.CreateFile(
            {
                'title': file_name, 
                'parents': [{'kind': 'drive#fileLink', 'id': parent_folder_id}]
            }
        )
        file.SetContentFile(local_file_path)
        file.Upload()
        return file['id']

    def download(self, local_file_path, target_path):
        target_id = self.path_to_id(target_path)
        file = self.drive.CreateFile({'id': target_id})
        file.GetContentFile(local_file_path)