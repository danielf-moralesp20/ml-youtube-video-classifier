import yaml


def read_yaml(stream):
    with open(stream) as conf_file:
        file = yaml.safe_load(conf_file)

    return file