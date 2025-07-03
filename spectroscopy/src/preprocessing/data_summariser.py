import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class DataSummariser:
    """
    Class for summarizing data through various visualizations and analyses.
    It includes methods for generating bar charts, correlation matrices, violin plots,
    boxplots, and more. It also provides functionality for detecting and removing outliers.
    """

    def __init__(self, parent_dir):
        """
        Initialize the DataSummariser with the parent directory.

        Parameters:
            parent_dir (str): The parent directory where results will be saved.
        """
        self.parent_dir = parent_dir
        self.title = None
        self.filepath = None
        self.fig = None
        self.ax = None
        self.outliers_indices_df = None

    def targets_nan_summary(
        self,
        targets_df: pd.DataFrame = None,
        leaf_state: str = None,
        season: int = None,
        figsize: tuple = (10, 6),
    ):
        """
        Plot a bar chart of the total number of missing values for each element.

        Parameters:
            targets_df (pd.DataFrame, optional): DataFrame containing the target data.
            leaf_state (str, optional): The state of the leaf ("dried", "fresh", or None). Default is None.
            season (int, optional): The season number (1, 2, 3, 4, or None). Default is None.
            figsize (tuple, optional): The size of the figure. Default is (10, 6).

        Returns:
            None
        """
        missing_values = targets_df.isnull().sum()
        self.title = f"Total Number of Missing Values ({DataSummariser._none_to_str(leaf_state).capitalize()}{DataSummariser._none_to_str(season)})"
        self.filepath = f"{self.parent_dir}\\data\\EDA_results\\bar_charts"
        self.fig, self.ax = plt.subplots(figsize=figsize)
        missing_values.plot(kind="bar", ax=self.ax)
        plt.title(self.title)
        plt.xlabel("Targets")
        plt.ylabel("No. of Missing Values")
        plt.show()

    def generate_stacked_barchart(
        self,
        df: pd.DataFrame = None,
        agg: str = None,
        category_col: str = None,
        stacks: list = None,
        figsize: tuple = (10, 6),
        colormap=["tab:orange", "tab:blue"],
        title: str = None,
    ):
        """
        Generate a stacked bar chart for the given DataFrame.

        Parameters:
            df (pd.DataFrame, optional): DataFrame containing the data to plot.
            agg (str, optional): The aggregation method for the y-axis. Default is None.
            category_col (str, optional): Column name for the categories on the x-axis. Default is None.
            stacks (list, optional): List of column names to be stacked. Default is None.
            figsize (tuple, optional): The size of the figure. Default is (10, 6).
            colormap (list, optional): List of colors for the stacks. Default is ['tab:orange', 'tab:blue'].
            title (str, optional): The title of the plot. Default is None.

        Returns:
            None
        """
        self.fig, self.ax = plt.subplots(figsize=figsize)
        self.title = title
        self.filepath = f"{self.parent_dir}\\data\\EDA_results\\bar_charts"
        df.plot(
            kind="bar",
            ax=self.ax,
            stacked=True,
            rot=0,
            edgecolor="black",
            color=colormap,
        )
        self.ax.set_ylabel(agg, fontweight="bold")
        self.ax.set_xlabel(category_col, fontweight="bold")
        self.ax.legend(stacks)
        plt.title(self.title)
        for c in self.ax.containers:
            labels = [int(v.get_height()) if v.get_height() > 0 else "" for v in c]
            self.ax.bar_label(
                c, labels=labels, label_type="center", fontweight="bold", color="black"
            )

    def generate_correlation_matrix(
        self,
        df: pd.DataFrame = None,
        method: str = "spearmen",
        leaf_state: str = None,
        season: int = None,
        filepath: str = None,
        mask: bool = False,
        threshold: int = 0.5,
        figsize: tuple = (10, 8),
    ):
        """
        Generate a correlation matrix heatmap for the given DataFrame.

        Parameters:
            df (pd.DataFrame, optional): DataFrame containing the data to plot.
            method (str, optional): The method to compute correlation ('pearson', 'kendall', 'spearman'). Default is "spearmen".
            leaf_state (str, optional): The state of the leaf ("dried", "fresh", or None). Default is None.
            season (int, optional): The season number (1, 2, 3, 4, or None). Default is None.
            filepath (str, optional): The path to save the plot. Default is None.
            mask (bool, optional): Whether to mask the correlation matrix. Default is False.
            threshold (int, optional): The threshold for masking. Default is 0.5.
            figsize (tuple, optional): The size of the figure. Default is (10, 8).

        Returns:
            None
        """
        spearman_corr = df.apply(pd.to_numeric, errors="coerce").corr(method=method)
        self.title = f"{method.title()} Correlation Matrix Heatmap {DataSummariser._none_to_str(leaf_state).capitalize()}{DataSummariser._none_to_str(season)}"
        if filepath is not None:
            self.filepath = filepath
        else:
            self.filepath = f"{self.parent_dir}\\data\\EDA_results\\correlation_matrix"
        self.fig, self.ax = plt.subplots(figsize=figsize)
        if mask is True:
            mask_df = np.abs(spearman_corr) < threshold
            sns.heatmap(
                spearman_corr,
                annot=True,
                cmap="coolwarm",
                mask=mask_df,
                center=0,
                ax=self.ax,
            )
        else:
            sns.heatmap(
                spearman_corr, annot=True, cmap="coolwarm", center=0, ax=self.ax
            )
        plt.title(self.title)
        plt.show()

    def save_graph(self, format: str = "pdf", dpi: int = 300, filepath: str = None):
        """
        Save the current graph to a file.

        Parameters:
            format (str, optional): The format to save the file in. Default is "pdf".
            dpi (int, optional): The resolution of the saved file. Default is 300.
            filepath (str, optional): Filepath to save the graph.

        Returns:
            None
        """
        if filepath is not None:
            self.filepath = filepath
        if not os.path.exists(self.filepath):
            os.makedirs(self.filepath)
        self.fig.savefig(
            f"{self.filepath}\\{self.title}.{format}", format=format, dpi=dpi
        )

    def generate_violin_plot(
        self,
        df: pd.DataFrame,
        x: str = None,
        y: str = None,
        hue: str = None,
        split: bool = False,
        scatter: bool = True,
        jitter: float = 0.1,
        leaf_state: str = None,
        season: int = None,
        figsize: tuple = (10, 8),
    ):
        """
        Generate a violin plot with optional scatter overlay.

        Parameters:
            df (pd.DataFrame): DataFrame containing the data to plot.
            x (str, optional): Column name for the x-axis. Default is None.
            y (str, optional): Column name for the y-axis. Default is None.
            hue (str, optional): Column name for the hue. Default is None.
            split (bool, optional): Whether to split the violin plot. Default is False.
            scatter (bool, optional): Whether to overlay a scatter plot. Default is True.
            jitter (float, optional): Amount of jitter for the scatter plot. Default is 0.1.
            leaf_state (str, optional): The state of the leaf ("dried", "fresh", or None). Default is None.
            season (int, optional): The season number (1, 2, 3, 4, or None). Default is None.
            figsize (tuple, optional): The size of the figure. Default is (10, 8).

        Returns:
            None
        """
        self.title = f"Violin plot ({DataSummariser._none_to_str(leaf_state).capitalize()}{DataSummariser._none_to_str(season)})"
        self.filepath = f"{self.parent_dir}\\data\\EDA_results\\violin_plots"
        self.fig, self.ax = plt.subplots(figsize=figsize)
        sns.violinplot(
            data=df,
            inner="quartile",
            x=x,
            y=y,
            hue=hue,
            palette="tab10",
            split=split,
            ax=self.ax,
        )
        if scatter is True:
            sns.stripplot(
                data=df,
                jitter=jitter,
                dodge=True,
                marker="o",
                alpha=0.5,
                color="k",
                ax=self.ax,
            )
        plt.title(self.title)
        plt.xlabel("Targets")
        plt.ylabel("Scaled Values")
        plt.show()

    def generate_boxplots(
        self,
        df: pd.DataFrame,
        x: str = None,
        y: str = None,
        hue: str = None,
        leaf_state: str = None,
        season: int = None,
        showfliers: bool = False,
        title: str = None,
        figsize: tuple = (10, 8),
    ):
        """
        Generate boxplots for the given DataFrame.

        Parameters:
            df (pd.DataFrame): DataFrame containing the data to plot.
            x (str, optional): Column name for the x-axis. Default is None.
            y (str, optional): Column name for the y-axis. Default is None.
            hue (str, optional): Column name for the hue. Default is None.
            leaf_state (str, optional): The state of the leaf ("dried", "fresh", or None). Default is None.
            season (int, optional): The season number (1, 2, 3, 4, or None). Default is None.
            showfliers (bool, optional): Whether to show outliers. Default is False.
            Title (str, optional): Title of the plot. Default is None.
            figsize (tuple, optional): The size of the figure. Default is (10, 8).

        Returns:
            None
        """
        if title is None:
            self.title = f"Boxplot ({DataSummariser._none_to_str(leaf_state).capitalize()}{DataSummariser._none_to_str(season)})"
        else:
            self.title = title
        self.filepath = f"{self.parent_dir}\\data\\EDA_results\\boxplots"
        self.fig, self.ax = plt.subplots(figsize=figsize)
        sns.boxplot(
            x=x,
            y=y,
            hue=hue,
            data=df,
            palette="tab10",
            showfliers=showfliers,
            ax=self.ax,
        )
        plt.title(self.title)
        plt.xlabel("Targets")
        plt.ylabel("Value")
        plt.legend(title=hue)
        plt.show()

    @staticmethod
    def features_nan_summary(
        leaf_df: pd.DataFrame = None, features_df: pd.DataFrame = None
    ) -> pd.DataFrame:
        """
        Plot a bar chart of the percentage of NaN values for each row in the
        reflectivity readings.Only rows with NaN percentage greater than 0 are included.

        Parameters:
            leaf_df (pd.DataFrame): DataFrame containing the leaf data.
            features_df (pd.DataFrame): DataFrame containing the features data.

        Returns:
            pd.DataFrame: A DataFrame with the sample_id, leaf_state, season, and NaN percentage.
        """
        nan_percentage_df = features_df.isnull().mean(axis=1) * 100
        nan_percentage_filtered_df = nan_percentage_df[(nan_percentage_df > 0)]
        features_nan_df = leaf_df.loc[
            nan_percentage_filtered_df.index, ["sample_id", "leaf_state", "season"]
        ]
        features_nan_df = features_nan_df.assign(
            nan_percentage=nan_percentage_filtered_df.values
        )
        return features_nan_df

    @staticmethod
    def summarize_skewness(df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Computes the skewness of each numerical column in the given DataFrame.

        Parameters:
            df (pd.DataFrame): Input DataFrame.

        Returns:
            pd.DataFrame: A summary DataFrame with skewness values and interpretation.
        """
        skewness_values = df.skew()
        skewness_summary = (
            pd.DataFrame(
                {
                    "Skewness": skewness_values,
                    "Interpretation": skewness_values.apply(
                        DataSummariser._interpret_skewness
                    ),
                }
            )
            .reset_index()
            .rename(columns={"index": "Column"})
        )

        return skewness_summary

    @staticmethod
    def _interpret_skewness(value):
        """
        Interpret the skewness value and return a descriptive string.

        Parameters:
            value (float): The skewness value.

        Returns:
            str: A descriptive string indicating the type of skewness.
        """
        if value < -1:
            return "Highly Negatively Skewed"
        elif -1 <= value < -0.5:
            return "Moderately Negatively Skewed"
        elif -0.5 <= value <= 0.5:
            return "Approximately Symmetric"
        elif 0.5 < value <= 1:
            return "Moderately Positively Skewed"
        else:
            return "Highly Positively Skewed"

    @staticmethod
    def min_max_scaler(df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Scale the features of the DataFrame to a range between 0 and 1 using Min-Max scaling.

        Parameters:
            df (pd.DataFrame, optional): DataFrame containing the data to be scaled.

        Returns:
            pd.DataFrame: A DataFrame with the scaled data.
        """
        scaler = MinMaxScaler()
        scaled_data = scaler.fit_transform(df)
        scaled_df = pd.DataFrame(scaled_data, columns=df.columns)
        return scaled_df

    @staticmethod
    def _none_to_str(value):
        """
        Convert None to an empty string, otherwise return the value as is.

        Parameters:
            value: The value to be converted.

        Returns:
            str: An empty string if the value is None, otherwise the value itself.
        """
        return "" if value is None else value

    def detect_outliers_iqr(self, df: pd.DataFrame = None) -> pd.DataFrame:
        """
        Identifies outliers in a DataFrame using the IQR method and returns a new DataFrame
        listing the indices of outliers for each column.

        Parameters:
            df (pd.DataFrame): Input DataFrame with numerical values.

        Returns:
            pd.DataFrame: A DataFrame where each column contains a list of indices of the outliers.
        """
        outlier_dict = {}

        for column in df.select_dtypes(include=["number"]).columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR

            outlier_indices = df[
                (df[column] < lower_bound) | (df[column] > upper_bound)
            ].index.tolist()
            outlier_dict[column] = outlier_indices

        self.outliers_indices_df = pd.DataFrame(
            dict([(k, pd.Series(v)) for k, v in outlier_dict.items()])
        )

        return self.outliers_indices_df

    def remove_outliers(
        self, targets_df: pd.DataFrame = None, count: int = 0, index_df: list = None
    ):
        """
        Remove outliers from the DataFrame based on the outlier indices.

        Parameters:
            targets_df (pd.DataFrame, optional): DataFrame containing the target data.
            count (int, optional): The threshold count for removing outliers. Default is 0.
            index_df (list, optional): List of indices to be removsed. Default is None.

        Returns:
            pd.DataFrame: DataFrame with the outliers removed.
        """
        unique_values = (
            self.outliers_indices_df.apply(lambda x: x.value_counts(dropna=True))
            .fillna(0)
            .astype(int)
        )
        indices_df = pd.DataFrame(
            unique_values.sum(axis=1).sort_values(ascending=False)
        )
        if not index_df:
            removed_outliers_df = targets_df.drop(
                list(indices_df.loc[indices_df[0] > count].index)
            )
        else:
            removed_outliers_df = targets_df.drop(index_df)
        return removed_outliers_df
