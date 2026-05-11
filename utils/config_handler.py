"""
yaml
k: v
"""

import os
import yaml
from utils.path_tool import get_abs_path


def _expand_env_vars(value):
    if isinstance(value, str):
        return os.path.expandvars(value)
    if isinstance(value, dict):
        return {key: _expand_env_vars(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_expand_env_vars(item) for item in value]
    if isinstance(value, tuple):
        return tuple(_expand_env_vars(item) for item in value)
    return value


def load_rag_config(config_path: str = get_abs_path("config/rag.yml"), encoding: str = "utf-8"):
    with open(config_path, "r", encoding=encoding) as f:
        return _expand_env_vars(yaml.safe_load(f) or {})


def load_chroma_config(config_path: str = get_abs_path("config/chroma.yml"), encoding: str = "utf-8"):
    with open(config_path,'r',encoding=encoding) as f:
        return _expand_env_vars(yaml.safe_load(f) or {})


def load_prompts_config(config_path: str = get_abs_path("config/prompts.yml"), encoding: str = "utf-8"):
    with open(config_path,'r',encoding=encoding) as f:
        return _expand_env_vars(yaml.safe_load(f) or {})


def load_agent_config(config_path: str = get_abs_path("config/agent.yml"), encoding: str = "utf-8"):
    with open(config_path,'r',encoding=encoding) as f:
        return _expand_env_vars(yaml.safe_load(f) or {})


rag_conf = load_rag_config()
chroma_conf = load_chroma_config()
prompts_conf = load_prompts_config()
agent_conf = load_agent_config()

if __name__ == '__main__':
    print(rag_conf["chat_model_name"])
