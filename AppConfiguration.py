import json


class AppConfiguration:
    __instance = None
    __config = None

    def __init__(self):
        if AppConfiguration.__instance != None:
            raise Exception("AppConfiguration é um singleton, use AppConfiguration.get_instance() para obtê-lo.")
        else:
            AppConfiguration.__instance = self
            self.__load_config()

    @staticmethod
    def get_instance():
        if AppConfiguration.__instance == None:
            AppConfiguration()
        return AppConfiguration.__instance

    def __load_config(self):
        with open("config.json") as f:
            AppConfiguration.__config = json.load(f)

    def get_config(self):
        return AppConfiguration.__config

    def set_config(self, config):
        AppConfiguration.__config = config