"""
utils/config.py

Config loading with command-line overrides, so experiments never need edited
copies of a YAML file:

    load_config("configs/eth_ucy.yaml", ["dataset.test_scene=hotel",
                                         "model.use_social=false"])

Override values are parsed as YAML (numbers, booleans, null, lists).
"""

import copy

import yaml


def apply_overrides(config: dict, overrides) -> dict:
    """Return a copy of config with dotted-path overrides ("a.b.c=value")."""
    config = copy.deepcopy(config)
    for item in overrides or []:
        if "=" not in item:
            raise ValueError(f"Override '{item}' must look like key.path=value")
        path, raw = item.split("=", 1)
        keys = path.strip().split(".")
        node = config
        for k in keys[:-1]:
            node = node.setdefault(k, {})
            if not isinstance(node, dict):
                raise ValueError(f"Override '{item}': '{k}' is not a section")
        node[keys[-1]] = yaml.safe_load(raw)
    return config


def load_config(path: str, overrides=None) -> dict:
    with open(path, "r", encoding="utf-8") as fh:
        config = yaml.safe_load(fh)
    return apply_overrides(config, overrides)
