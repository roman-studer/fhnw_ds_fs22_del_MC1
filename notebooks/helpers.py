import yaml

def remove_suffix(input_string, suffix):
    if suffix and input_string.endswith(suffix):
        return input_string[:-len(suffix)]
    return input_string


def get_config():
    with open('../config.yml') as f:
        config = yaml.load(f, Loader=yaml.BaseLoader)
    return config