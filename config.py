import configparser

config = configparser.ConfigParser()

config.read("config.ini")


def get_classes():
    return config["main"]["classes"].split(",")


config.get_classes = get_classes
