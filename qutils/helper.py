import yaml

def parse_yaml_config(yaml_file:str, printOut=False):
    """
    Function to parse the configuration file and return the parameters.
    """
    # Load vars.yaml
    with open(yaml_file, "r") as f:
        config = yaml.safe_load(f)
    if printOut:
        for key, value in config.items():
            print(f"{key}: {value}")

    return config
