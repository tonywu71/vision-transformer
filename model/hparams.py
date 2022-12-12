import yaml

CONFIG_FILEPATH = "config.yaml"

def read_config() -> dict:
    with open(CONFIG_FILEPATH, "r") as f:
        config = yaml.safe_load(f)
    
    config["learning_rate"] = float(config["learning_rate"])
    config["input_shape"] = tuple(config["input_shape"])
    config["num_patches"] = (config["image_size"] // config["patch_size"]) ** 2
    
    return config
