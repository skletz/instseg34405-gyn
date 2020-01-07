import configparser
import json
import os


class ConfigParser(object):

    @staticmethod
    def parser_args(conf_file):
        config_parser = configparser.ConfigParser()
        config_parser.read(conf_file)
        conf = Config(config_parser._sections)
        return conf


class Config(object):

    def __init__(self, confs):
        conf_dict = self.to_dict(confs)

        for key, value in conf_dict.items():
            conf_dict[key] = Params(dict=value)

        self.__dict__.update(conf_dict)

    def as_dict(self):
        return self.d

    def to_dict(self, config):
        """
        Nested OrderedDict to normal dict.
        Cast types: currently supported: int, float, list, bool, expressions
        """
        d = json.loads(json.dumps(config))
        d = TypeHandler.cast_dtypes(d)
        d = dict((k.lower(), v) for k, v in d.items())
        return d

    def save(self, path):
        config = configparser.ConfigParser()
        for key1, data1 in self.dict.items():
            config[key1] = {}
            for key2, data2 in data1.dict.items():
                config[key1][key2] = str(data2)

        with open(path, 'w') as configfile:
            config.write(configfile)
            configfile.flush()

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class Params(object):

    def __init__(self, dict):
        self.__dict__.update(dict)

    def save(self, json_path):
        with open(json_path, 'w') as f:
            json.dump(self.__dict__, f, indent=4)

    def update(self, json_path):
        with open(json_path) as f:
            params = json.load(f)
            self.__dict__.update(params)

    @property
    def dict(self):
        """Gives dict-like access to Params instance by `params.dict['learning_rate']"""
        return self.__dict__


class TypeHandler:

    @staticmethod
    def cast_dtypes(tmp):
        d = tmp.copy()
        for key, value in d.items():
            if isinstance(value, str):
                s = d[key]

                if TypeHandler.is_int(s):
                    if "." in s:
                        d[key] = float(s)
                    else:
                        d[key] = int(s)
                elif TypeHandler.is_float(s):
                    if "." in s or "e" in s:
                        d[key] = float(s)
                elif TypeHandler.is_expression(s):
                    d[key] = eval(s)
                elif TypeHandler.is_bool(s):
                    if s == "True" or s == "true":
                        d[key] = True
                    else:
                        d[key] = False
                elif TypeHandler.is_list(s):
                    s = s.strip("[")
                    s = s.strip("]")
                    list_values = s.split(",")
                    converted = []
                    for i in list_values:
                        # convert into string and remove whitespaces
                        converted.append(str(i).strip())
                    d[key] = converted
                else:
                    if s.startswith('${'):
                        d[key] = os.path.expandvars(value)

            if isinstance(value, dict):
                d[key] = TypeHandler.cast_dtypes(value)

        return d

    @staticmethod
    def is_list(x):
        if x.startswith("["):
            return True
        return False

    @staticmethod
    def is_bool(x):
        if x == "True" or x == "False" or x == "true" or x == "false":
            return True
        else:
            return False

    @staticmethod
    def is_expression(x):
        try:
            a = eval(x)
        except NameError:
            return False
        except SyntaxError:
            return False
        else:
            return True

    @staticmethod
    def is_float(x):
        try:
            a = float(x)
        except ValueError:
            return False
        else:
            return True

    @staticmethod
    def is_int(x):
        try:
            a = float(x)
            b = int(a)
        except ValueError:
            return False
        else:
            return a == b
