import json

def read_conf(json_path):
    with open (json_path) as json_file:
        config=json.load(json_file)
    return config