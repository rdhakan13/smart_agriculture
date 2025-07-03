import pandas as pd
import re
import os
import logging
from spectroscopy.src.common.constants import SAMPLES_COLUMN_MAPPING, ELEMENT_COLUMNS
from spectroscopy.src.common.utility_functions import get_feature_columns


class LeafSampleReader:
    """
    Class for reading and processing leaf sample CSV files.
    It can filter files based on leaf state and season, and extract features and
    targets from the data.
    """

    def __init__(self, folder_path, file_extension=".csv"):
        """
        Initialize the LeafSampleReader with the folder path and file extension.

        Parameters:
            folder_path (str): Path to the folder containing CSV files.
            file_extension (str): File extension of the CSV files (default is ".csv").
        """
        self.folder_path = folder_path
        self.file_extension = file_extension

    def read_all_csvs(self, leaf_state: str = None, season: int = None) -> pd.DataFrame:
        """
        Read in all csvs relating to the leaf_state/season selection to a pandas dataframe.
        Leaving one of the selections blank will read all csvs of that option.
        Adds columns to specify the leaf state and season.

        Parameters:
            leaf_state (str): Leaf state to filter by (e.g., "FRESH", "DRIED").
            season (int): Season to filter by (e.g., 1, 2, 3, 4).

        Returns:
            pd.DataFrame: Combined dataframe of all selected CSV files.
        """
        csv_files = self._get_csv_file_paths(
            leaf_states=[leaf_state] if leaf_state else None,
            seasons=[season] if season else None,
        )
        return self._read_csv_files(csv_files)

    def read_selected_csvs(
        self, leaf_states: list[str] = None, seasons: list[int] = None
    ) -> pd.DataFrame:
        """
        Read in selected CSV files filtered by leaf_states and seasons into a pandas dataframe.
        If no specific filters are applied, all CSVs are read.

        Parameters:
            leaf_states (list[str]): List of leaf states to filter by (e.g., ["FRESH", "DRIED"]).
            seasons (list[int]): List of seasons to filter by (e.g., [1, 2, 3, 4]).

        Returns:
            pd.DataFrame: Combined dataframe of all selected CSV files.
        """
        csv_files = self._get_csv_file_paths(leaf_states, seasons)
        return self._read_csv_files(csv_files)

    def _read_csv_files(self, csv_files: list[str]) -> pd.DataFrame:
        """
        Read in CSV files and combine them into a single dataframe.

        Parameters:
            csv_files (list[str]): List of CSV file paths to read.

        Returns:
            pd.DataFrame: Combined dataframe of all selected CSV files.
        """
        if not csv_files:
            raise FileNotFoundError(
                "No matching CSV files found in the specified folder."
            )

        dataframes = []
        for file in csv_files:
            logging.info(f"Reading {file}")
            try:
                leaf_state = self._extract_leaf_state(file)
                season = self._extract_season(file)

                df = pd.read_csv(file)
                df = self._rename_column_names(df)
                df["leaf_state"] = leaf_state
                df["season"] = season
                dataframes.append(df)
            except Exception as e:
                print(f"Error reading {file}: {e}")

        if dataframes:
            combined_df = pd.concat(dataframes, ignore_index=True)
            return combined_df
        else:
            raise ValueError("No valid CSV files could be read.")

    def _get_csv_file_paths(
        self, leaf_states: list[str] = None, seasons: list[int] = None
    ) -> list[str]:
        """
        Get file paths of CSV files filtered by leaf_states and seasons.
        If no filters are provided, all CSV files are returned.

        Parameters:
            leaf_states (list[str]): List of leaf states to filter by (e.g., ["FRESH", "DRIED"]).
            seasons (list[int]): List of seasons to filter by (e.g., [1, 2, 3, 4]).

        Returns:
            list[str]: List of filtered file paths.
        """
        filtered_files = []
        for file in os.listdir(self.folder_path):
            if file.endswith(self.file_extension):
                if leaf_states and not any(
                    state.upper() in file.upper() for state in leaf_states
                ):
                    continue
                if seasons:
                    match = re.search(r"season(\d+)", file, re.IGNORECASE)
                    if not match or int(match.group(1)) not in seasons:
                        continue
                filtered_files.append(os.path.join(self.folder_path, file))
        return filtered_files

    @staticmethod
    def extract_targets(df: pd.DataFrame = None):
        """
        Extracts target columns from the dataframe.

        Parameters:
            df (pd.DataFrame): Dataframe containing the data.

        Returns:
            pd.DataFrame: Dataframe containing only the target columns.
        """
        targets = [col for col in ELEMENT_COLUMNS if col in df.columns]
        return df.loc[:, targets] if targets else pd.DataFrame()

    @staticmethod
    def extract_features(df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Extracts feature columns from the dataframe.

        Parameters:
            df (pd.DataFrame): Dataframe containing the data.

        Returns:
            pd.DataFrame: Dataframe containing only the feature columns.
        """
        return df[get_feature_columns(df)]

    @staticmethod
    def _rename_column_names(df):
        """
        Renames the columns of the dataframe based on a predefined mapping.
        This is useful for ensuring consistent column names across different datasets.

        Parameters:
            df (pd.DataFrame): Dataframe containing the data.

        Returns:
            pd.DataFrame: Dataframe with renamed columns.
        """
        df.rename(columns=SAMPLES_COLUMN_MAPPING, inplace=True)
        return df

    @staticmethod
    def _extract_leaf_state(filename: str) -> str:
        """
        Extracts the leaf state from the filename.

        Parameters:
            filename (str): Name of the file.

        Returns:
            str: Leaf state extracted from the filename.
        """
        return (
            "DRIED"
            if "DRIED" in filename.upper()
            else "FRESH"
            if "FRESH" in filename.upper()
            else "UNKNOWN"
        )

    @staticmethod
    def _extract_season(filename: str) -> int:
        """
        Extracts the season from the filename.

        Parameters:
            filename (str): Name of the file.

        Returns:
            int: Season extracted from the filename.
        """
        match = re.search(r"season(\d+)", filename, re.IGNORECASE)
        return int(match.group(1)) if match else -1
