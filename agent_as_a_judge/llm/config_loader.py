import yaml
import os

def load_llm_config(config_path="llm_config.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    for model, params in config["models"].items():
        for k, v in params.items():
            if isinstance(v, str) and v.startswith("${") and v.endswith("}"):
                env_key = v[2:-1]
                config["models"][model][k] = os.getenv(env_key, "")
    return config 