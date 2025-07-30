import os
import argparse
from copy import deepcopy
from typing import Union, Optional
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig


def add_args_from_config(config, parser, prefix=""):
    r"""Add new arguments to an argparser by using a predefined configs. e.g.,
    config = {'a': {'b': 123}}, access the config['a']['b'] by
    `python custom.py --a-b 234`.
    """
    for key, value in config.items():
        # '--' For the args under the root
        arg_name = f"-{prefix}-{key}"
        # Add args recursively if cfg is a nested DictConfig
        if OmegaConf.is_dict(value):
            add_args_from_config(value, parser, prefix=f"{prefix}-{key}")
        else:
            if OmegaConf.is_list(value):
                value = OmegaConf.to_container(value)
                parser.add_argument(
                    arg_name, type=type(value[0]), nargs="+", default=None
                )
                continue

            arg_type = type(value)
            if arg_type == bool:
                parser.add_argument(
                    arg_name,
                    action="store_false" if value else "store_true",
                    default=None,
                )
            elif arg_type == type(None):
                parser.add_argument(
                    arg_name, default=None,
                )
            else:
                parser.add_argument(
                    arg_name, type=arg_type, default=None,
                ) 


def update_config_from_args(config, args):
    r"""Update an existing config by using a set of arguments.
    The arguments should be created by `add_args_from_config`.
    """

    def _recur_update_cfgs_from_args(config, args, prefix=""):
        cur_config = deepcopy(config)
        for key in config:
            if OmegaConf.is_dict(config[key]):
                updated_cfgs = _recur_update_cfgs_from_args(
                    config[key], args, prefix=f"{prefix}-{key}"
                )
                cur_config = OmegaConf.merge(cur_config, {key: updated_cfgs})
            else:
                arg_name = f"{prefix}-{key}".lstrip("-").replace("-", "_")
                if hasattr(args, arg_name):
                    override_v = getattr(args, arg_name)
                    cur_config[key] = (
                        override_v if override_v is not None else config[key]
                    )
        return cur_config

    # Update config from each subgroup
    for k, v in config["__subgroup__"].items():
        sg_cfgs_path = getattr(args, f"__subgroup__-{k}".replace("-", "_"))
        if sg_cfgs_path is not None:
            updated_sg_cfgs = load_config(sg_cfgs_path)
            config = OmegaConf.merge(config, {k: updated_sg_cfgs})
    del config.__subgroup__

    # Update config from each leaf node
    config = _recur_update_cfgs_from_args(config, args, prefix="")
    return config


def load_config(
    config_path: Union[dict, str, DictConfig], dump_path: Optional[str] = None
) -> dict:
    r"""Load config from yaml file.
    This function will also read the yaml files
    if they are specified in '__subgroup__'. e.g.,
    [within `config_path`]
        __subgroup__:
            a: path_to_yaml_a
            b: path_to_yaml_b
            ...
        attribute 1:
            ...
    ------
    RETURNS: OmegaConf.DictConfig
    """
    if isinstance(config_path, str):
        with open(config_path, "r") as file:
            config = OmegaConf.load(file)
    elif isinstance(config_path, dict):
        config = OmegaConf.create(config_path)
    else:
        assert OmegaConf.is_config(
            config_path
        ), f"config_path must be config path, dict, or DictConfig"
        config = config_path

    if "__subgroup__" in config:
        subgroups = config.get("__subgroup__")
        cur_cfg_dir = os.path.dirname(os.path.abspath(config_path))
        for sg_name, sg_config_path in subgroups.items():
            sg_abs_pth = os.path.join(cur_cfg_dir, sg_config_path)
            sg_config = OmegaConf.load(sg_abs_pth)
            config = OmegaConf.merge(config, {sg_name: sg_config})
            config.__subgroup__[sg_name] = sg_abs_pth  # update sub cfg path

    return config


def dynamic_config(description: Optional[str] = None, verbose: bool = True):
    r"""Load configuration from both yaml file and command line.
    The config in the yaml will be overrided by the arg passed from command line.
    e.g.,
        [Command line] python3 custom.py --config_path /path/to/config.yaml --a-b-c=123
        [Python  file] cfgs = dynamic_config('A demo for dynamic configuration.')
                       cfgs.to_yaml('path/to/output/config.yaml')  # log the config of this trial
    ------
    RETURNS:
        DictConfig.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("--config_path", type=str, help="Path to the yaml file.")
    # Get predefined configs and add new args dynamically
    args, remaining_args = parser.parse_known_args()
    cfgs = load_config(args.config_path)
    add_args_from_config(cfgs, parser)
    # Override values in `cfgs` if applicable
    args = parser.parse_args(remaining_args)
    cfgs = update_config_from_args(cfgs, args)

    if verbose:
        import logging

        log = logging.getLogger(__name__)
        log.info(f"Successfully setup the configuration:\n{OmegaConf.to_yaml(cfgs)}")

    return cfgs


def dump_config(cfgs, dump_path):
    dump_dir = os.path.dirname(os.path.abspath(dump_path))
    os.makedirs(dump_dir, exist_ok=True)
    with open(dump_path, "w") as file:
        OmegaConf.save(cfgs, f=file)


if __name__ == "__main__":
    cfgs = dynamic_config()
    print("Updated Configuration:")
    print(OmegaConf.to_yaml(cfgs))
    import ipdb

    ipdb.set_trace()
