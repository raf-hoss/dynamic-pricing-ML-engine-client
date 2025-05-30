import os
import yaml
import logging
import pandas as pd

# Create logger
def setup_logger(name: str = "dynamic_pricing", log_file: str = "logs/pipeline_logs.log"):
    os.makedirs(os.path.dirname(log_file), exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)

    fh = logging.FileHandler(log_file)
    formatter = logging.Formatter('%(asctime)s — %(levelname)s — %(message)s')
    fh.setFormatter(formatter)

    if not logger.handlers:
        logger.addHandler(fh)

    return logger

# Create directory if doesn't exist
def ensure_dir(path: str):
    os.makedirs(path, exist_ok=True)

# Load YAML config
def load_yaml_config(path: str = "src/config/config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)

# Print quick summary of a DataFrame
def summarize_df(df: pd.DataFrame, name: str = "DataFrame"):
    print(f"\n📊 Summary of {name}:")
    print(f"Shape: {df.shape}")
    print("Columns:", df.columns.tolist())
    print("Missing values:\n", df.isnull().sum())
    print(df.describe(include="all").T)