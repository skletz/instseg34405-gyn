import os
import fnmatch
import numpy as np
import json


class IOUtil:

    @staticmethod
    def create_dir(dir):
        if not os.path.exists(dir):
            os.makedirs(dir)

        return dir

    @staticmethod
    def join_dir(dir1, dir2):
        return os.path.join(dir1, dir2)

    @staticmethod
    def get_files(data_dir, pattern="*", ignore=[]):
        """
        Iterates recursively over a given directory and collects all files with the given patter
        :param data_dir: (string) directory to search
        :param pattern: (regex)match the filename
        :return: dataloader
        """
        for dir_paths, dir_names, filenames in os.walk(data_dir):
            for file_name in fnmatch.filter(filenames, pattern):

                if not any(x in file_name for x in ignore):
                    if not file_name.startswith('.'):
                        yield os.path.join(dir_paths, file_name)

    @staticmethod
    def convert_to_valid_json(_dict):
        for key, value in _dict.items():

            if isinstance(value, dict):
                for key2, value2 in value.items():
                    if isinstance(value2, (np.ndarray, np.generic)):
                        _dict[key][key2] = value2.tolist()
            elif isinstance(value, (np.ndarray, np.generic)):
                _dict[key] = value.tolist()
            else:
                _dict[key] = value

        return _dict

    @staticmethod
    def save_dict_to_json(_dict, json_path):
        """
        Saves dict in json file
        :param _dict: (dict) of values
        :param json_path: (string) path to json file
        """

        with open(json_path, 'w') as f:
            json.dump(_dict, f, indent=4)
