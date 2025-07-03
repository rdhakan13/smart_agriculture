import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from spectroscopy import LeafSampleReader
from spectroscopy.src.common.utility_functions import get_working_directory


class DataReducer:
    """
    Class for reducing data dimensionality using PCA or binning methods.
    """

    def __init__(self, method: str):
        """
        Initializes the DataReducer class.
        """
        self.reader = LeafSampleReader(f"{get_working_directory()}/data/leaf_samples")
        self.method = method
        self.reduced_data = None
        self.reduced_features = None
        self.explained_variance = None
        self.cumulative_variance = None

    def reduce_data(
        self,
        df: pd.DataFrame,
        n_components: int = None,
        wavelength_bins: int = 10,
        random_state: int = 42,
        reset_index=False,
    ) -> pd.DataFrame:
        """
        Reduces the data dimensionality using PCA or binning methods.

        Parameters:
            df (pd.DataFrame): The dataframe to reduce.
            n_components (int): Number of principal components to keep (for PCA).
            wavelength_bins (int): Number of wavelength bins (for binning).
            random_state (int): Random state for reproducibility.
            reset_index (bool): Whether to reset the index of the dataframe.

        Returns:
            pd.DataFrame: The reduced dataframe.
        """
        if reset_index:
            df = df.reset_index(drop=True)
        if self.method == "pca":
            self._conduct_pca(
                df=df, n_components=n_components, random_state=random_state
            )
        elif self.method == "binning":
            self._conduct_binning(df=df, wavelength_bins=wavelength_bins)
        elif self.method is None:
            self.reduced_data = df
        else:
            raise ValueError(f"Unknown data reduction method: {self.method}")
        return self.reduced_data

    def _conduct_pca(
        self, df: pd.DataFrame, n_components: int = None, random_state: int = 42
    ) -> pd.DataFrame:
        """
        Conducts PCA on the given dataframe.

        Parameters:
            df (pd.DataFrame): The dataframe to reduce.
            n_components (int): Number of principal components to keep.
            random_state (int): Random state for reproducibility.

        Returns:
            pd.DataFrame: The reduced dataframe.
        """
        features_df = self.reader.extract_features(df)
        pca = PCA(n_components=n_components, random_state=random_state)
        self.reduced_features = pca.fit_transform(features_df)
        self.reduced_features = pd.DataFrame(
            self.reduced_features,
            columns=[f"PC{i + 1}" for i in range(pca.n_components_)],
        )
        self.explained_variance = pca.explained_variance_ratio_
        self.cumulative_variance = np.cumsum(self.explained_variance)
        self.reduced_data = df.drop(columns=features_df.columns).join(
            pd.DataFrame(self.reduced_features)
        )

    def generate_scree_plot(self):
        """
        Generates a scree plot showing the explained variance ratio of each principal component.

        Parameters:
            None

        Returns:
            None
        """
        plt.figure(figsize=(8, 5))
        plt.bar(
            range(1, len(self.explained_variance) + 1),
            self.explained_variance,
            alpha=0.7,
            label="Individual Explained Variance",
        )
        plt.plot(
            range(1, len(self.explained_variance) + 1),
            self.explained_variance,
            marker="o",
            linestyle="dashed",
            color="r",
            label="Cumulative Explained Variance",
        )
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Explained Variance Ratio")
        plt.title("Scree Plot")
        plt.legend()
        plt.grid()
        plt.show()

    def generate_cummulative_variance_plot(self):
        """
        Generates a cumulative variance plot showing the cumulative explained variance.

        Parameters:
            None

        Returns:
            None
        """
        plt.figure(figsize=(8, 5))
        plt.plot(
            range(1, len(self.cumulative_variance) + 1),
            self.cumulative_variance,
            marker="o",
            linestyle="dashed",
            color="b",
        )
        plt.axhline(
            y=0.95, color="r", linestyle="dashed", label="95% Variance Explained"
        )
        plt.xlabel("Number of Principal Components")
        plt.ylabel("Cumulative Explained Variance")
        plt.title("Cumulative Explained Variance")
        plt.legend()
        plt.grid()
        plt.show()

    def generate_pca_plot(self):
        """
        Generates a PCA biplot showing the first two principal components.

        Parameters:
            None

        Returns:
            None
        """
        plt.figure(figsize=(8, 5))
        plt.scatter(self.reduced_features[:, 0], self.reduced_features[:, 1], alpha=0.7)
        plt.xlabel("Principal Component 1")
        plt.ylabel("Principal Component 2")
        plt.title("PCA Biplot")
        plt.grid()
        plt.show()

    def _conduct_binning(self, df: pd.DataFrame, wavelength_bins: int = 10):
        """
        Conducts binning on the given dataframe.

        Parameters:
            df (pd.DataFrame): The dataframe to reduce.
            wavelength_bins (int): Number of wavelength bins.

        Returns:
            None
        """
        features_df = self.reader.extract_features(df)
        features_df = self._convert_column_names_to_numbers(features_df)
        new_cols = list(features_df.columns)
        delta = new_cols[1] - new_cols[0]
        increment = (wavelength_bins / delta) + 1
        self.reduced_features = {}
        for i in range(0, len(new_cols), int(increment)):
            start = new_cols[i]
            end = start + wavelength_bins
            if end in new_cols:
                self.reduced_features[f"{start}-{end}"] = features_df[
                    [start, end]
                ].mean(axis=1)
        self.reduced_features = pd.DataFrame(self.reduced_features)
        self.reduced_data = df.drop(
            columns=self.reader.extract_features(df).columns
        ).join(pd.DataFrame(self.reduced_features))

    @staticmethod
    def _convert_column_names_to_numbers(df: pd.DataFrame) -> pd.DataFrame:
        """
        Converts column names from wavelength format (e.g., "400nm") to numeric format (e.g., 400.0).

        Parameters:
            df (pd.DataFrame): The dataframe with wavelength columns.

        Returns:
            pd.DataFrame: The dataframe with numeric column names.
        """
        pattern = re.compile(r"^\d+(\.\d+)?nm$")
        new_column_names = {
            col: float(col.replace("nm", ""))
            for col in df.columns
            if pattern.match(col)
        }
        df = df.rename(columns=new_column_names)
        return df
