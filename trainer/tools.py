import yaml


class YamlConfig(object):

    @staticmethod
    def load(filepath: str):
        with open(filepath) as filestream:
            config = yaml.safe_load(filestream)
        return config
