import re
import os
import yaml
import sklearn
import datetime
import logging
import pandas as pd
from pathlib import Path


def get_working_directory() -> str:
    """
    Returns the working directory of the script.
    It traverses up the directory tree until it finds a directory containing a .git folder.

    Parameters:
        None
    """
    current_path = Path(__file__).resolve()
    while current_path != current_path.parent:
        if (current_path / ".git").exists():
            break
        current_path = current_path.parent
    return current_path


def get_feature_columns(df: pd.DataFrame) -> list:
    """
    Extracts a list of column names that match the format '200nm', '400nm', etc.

    Parameters:
        df (pd.DataFrame): The dataframe to extract feature columns from.

    Returns:
        list: A list of column names that match the specified format.
    """
    pattern = re.compile(r"^\d+(\.\d+)?nm$")
    reduced_pattern = re.compile(r"PC\d+")
    return [
        col for col in df.columns if pattern.match(col) or reduced_pattern.match(col)
    ]


def train_test_split(
    df: pd.DataFrame, method: str = "stratified", test_season: int = 4
) -> tuple:
    """
    Splits the dataframe into training and testing sets based on the specified method.

    Parameters:
        df (pd.DataFrame): The dataframe to split.
        method (str): The method for splitting. Options are "stratified" or "season_based".
        test_season (int): The season to use for the "season_based" method.

    Returns:
        tuple: A tuple containing the training and testing dataframes.
    """
    training_df = None
    testing_df = None
    if method == "stratified":
        training_df, testing_df = sklearn.model_selection.train_test_split(
            df, test_size=0.2, random_state=42, stratify=df["season"]
        )
    elif method == "season_based":
        training_df = df.loc[df["season"] != test_season]
        testing_df = df.loc[df["season"] == test_season]
    return training_df, testing_df


def load_config(config_filename: str) -> dict:
    """
    Loads a YAML configuration file from the configs directory.

    Parameters:
        config_filename (str): The name of the configuration file to load.

    Returns:
        dict: The loaded configuration as a dictionary.
    """
    config_path = f"{get_working_directory()}\\configs\\{config_filename}"
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found at path: {config_path}")
    else:
        with open(config_path, "r") as file:
            config = yaml.safe_load(file)
    return config


def create_timestamp_filename(prefix="data", suffix=".csv") -> str:
    """
    Create a filename with a timestamp.

    Parameters:
        prefix (str): The prefix for the filename.
        suffix (str): The suffix for the filename.

    Returns:
        str: The generated filename with a timestamp.
    """
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M%S")  # Format: YYYYMMDD_HHMMSS
    filename = f"{prefix}_{timestamp}{suffix}"
    return filename


def get_log_level(level_str) -> int:
    """
    Convert a string log level to the corresponding logging module level.

    Parameters:
        level_str (str): The string representation of the log level.

    Returns:
        int: The corresponding logging level constant.
    """
    level_mapping = {
        "DEBUG": logging.DEBUG,
        "INFO": logging.INFO,
        "WARNING": logging.WARNING,
        "ERROR": logging.ERROR,
        "CRITICAL": logging.CRITICAL,
    }
    return level_mapping.get(level_str.upper(), logging.INFO)
