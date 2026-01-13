import yaml
import argparse
from types import SimpleNamespace


def load_config(config_path):
    with open(config_path, "r") as f:
        config_dict = yaml.safe_load(f)

    def dict_to_obj(d):
        if not isinstance(d, dict):
            return d
        return SimpleNamespace(**{k: dict_to_obj(v) for k, v in d.items()})

    return dict_to_obj(config_dict)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, default="configs/default.yaml")

    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()
    return args
