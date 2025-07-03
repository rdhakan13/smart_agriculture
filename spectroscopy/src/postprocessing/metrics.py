import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_squared_error,
    root_mean_squared_error,
    r2_score,
    mean_absolute_error,
    mean_absolute_percentage_error,
)
from spectroscopy.src.common.constants import ELEMENT_UNITS


class Metrics:
    """
    Class to calculate metrics for model performance evaluation.
    It can handle multiple targets and can be extended to include additional metrics.
    The class can also handle different leaf states and seasons for the data.
    """

    def __init__(
        self,
        y_pred,
        y_test,
        leaf_state: str = None,
        season: int = None,
        targets: list = None,
        metrics: list = None,
        weights: dict = None,
    ):
        """
        Class can be initiated with or without leaf_state and season. If none 
        given, it assumes it doesn't need to separate in the indices of the resulting
        dataframe. If no metric is given,it will calculate for all.

        Parameters:
            y_pred (pd.DataFrame or np.ndarray): Predicted values.
            y_test (pd.DataFrame or np.ndarray): Actual values.
            leaf_state (str): Leaf state for the data.
            season (int): Season for the data.
            targets (list): List of target names to calculate metrics for.
            metrics (list): List of metrics to calculate. Default is None, which calculates all.
            weights (dict): Weights for each target, used for weighted RMSE calculation.

        Returns:
            None
        """
        if y_pred.shape != y_test.shape:
            raise ValueError(
                f"y_pred {y_pred.shape} and y_test {y_test.shape} must have the same shape. Check to ensure the actual and predicted targets are aligned."
            )

        self.y_pred = Metrics._convert_df(y_pred, y_test.columns.tolist())

        if targets is not None:
            try:
                self.y_pred = self.y_pred[targets]
                self.y_test = y_test[targets]
            except:
                raise ValueError(
                    "List of targets provided, but error slicing y_test with them. Check they exist in the test set."
                )
        else:
            self.y_test = y_test

        # Null columns need dropping prior to any further processing
        columns_to_drop = []
        for col in y_test.columns:
            if y_test[col].isnull().all():
                columns_to_drop.append(col)

        self.y_test = self.y_test.drop(columns_to_drop, axis=1)
        self.y_pred = self.y_pred.drop(columns_to_drop, axis=1)
        self.target_names = self.y_test.columns.tolist()
        self.leaf_state = leaf_state
        self.season = season

        if metrics is None:
            self.metrics = ["r2", "mse", "rmse", "rrmse", "mae", "mape"]
            if weights is not None:
                self.metrics += ["wrmse"]
        else:
            self.metrics = metrics

        if "wrmse" in self.metrics and weights is None:
            print(
                "Weights is not specified, wRMSE ignored. Specify weights in order to calculate."
            )
            self.metrics.remove("wrmse")
        elif "wrmse" in self.metrics and weights is not None:
            self.weights = [
                weights[target] for target in self.target_names if target in weights
            ]

        if leaf_state is None or season is None:
            self.results = pd.DataFrame(
                columns=self.metrics, index=self.target_names, dtype="float"
            )
            self.results.index.name = "target"
        else:
            index = pd.MultiIndex.from_product(
                [[leaf_state], [season], self.target_names],
                names=["leaf_state", "season", "target"],
            )
            self.results = pd.DataFrame(
                columns=self.metrics, index=index, dtype="float"
            )

        self._calc_single_metrics()

    @staticmethod
    def _convert_df(x, columns: list) -> pd.DataFrame:
        """
        Converts the input to a DataFrame with specified columns.
        If the input is already a DataFrame, it will be returned as is.
        If it is a numpy array, it will be converted to a DataFrame.

        Parameters:
            x (pd.DataFrame or np.ndarray): Input data to be converted.
            columns (list): List of column names for the DataFrame.

        Returns:
            pd.DataFrame: Converted DataFrame with specified columns.
        """
        if isinstance(x, np.ndarray):
            return pd.DataFrame(x, columns=columns, dtype="float")
        else:
            return x

    def _calc_single_metrics(self):
        """
        Main method to calculate all metrics selected.

        Parameters:
            None

        Returns:
            None
        """
        for metric in self.metrics:
            scores = self._calc_single_metric(metric)
            self.results = self._fill_metrics(
                self.results,
                scores,
                self.leaf_state,
                self.season,
                self.target_names,
                metric,
            )

    def add_dataset(self, y_pred, y_test, leaf_state=None, season=None):
        """
        This allows for the addition of a new dataset to the existing metrics object.
        It will not overwrite the existing data, but will add to it. This is useful for
        aggregating results from multiple datasets.

        Parameters:
            y_pred (pd.DataFrame or np.ndarray): Predicted values.
            y_test (pd.DataFrame or np.ndarray): Actual values.
            leaf_state (str): Leaf state for the data.
            season (int): Season for the data.

        Returns:
            None
        """
        self.target_names = y_test.columns.tolist()
        self.y_pred = Metrics._convert_df(y_pred, self.target_names)
        self.y_test = y_test
        self.leaf_state = leaf_state
        self.season = season
        index = self._extend_index(
            self.results.index, self.leaf_state, self.season, self.target_names
        )
        self.results = self.results.reindex(index)
        self._calc_single_metrics()

    @staticmethod
    def _extend_index(
        idx: pd.DataFrame.index, leaf_state: object, season: int, target: list
    ) -> pd.DataFrame.index:
        """
        The method of concatenation values is to first extend the indices, 
        which allows for df.loc to be used in _fill_metrics to populate values.

        Parameters:
            idx (pd.Index): Existing index of the DataFrame.
            leaf_state (object): Leaf state for the data.
            season (int): Season for the data.
            target (list): List of target names to calculate metrics for.

        Returns:
            pd.Index: New index with extended values.
        """
        if leaf_state is None or season is None:
            return idx.union(pd.Index([target]))
        else:
            new_idx = pd.MultiIndex.from_product(
                [[leaf_state], [season], target],
                names=["leaf_state", "season", "target"],
            )
            return idx.union(new_idx)

    @staticmethod
    def _fill_metrics(
        df: pd.DataFrame,
        scores: list,
        leaf_state: object,
        season: int,
        targets: list,
        metric: object,
    ) -> pd.DataFrame:
        """
        This populates the dataframe iteratively with the scores once the
        dataframe has been reindexed.

        Parameters:
            df (pd.DataFrame): DataFrame to be populated.
            scores (list): List of scores to populate the DataFrame with.
            leaf_state (object): Leaf state for the data.
            season (int): Season for the data.
            targets (list): List of target names to calculate metrics for.
            metric (object): Metric name to be used as column header.

        Returns:
            pd.DataFrame: DataFrame with populated scores.
        """
        new_df = df.copy()
        for i, t in enumerate(targets):
            if leaf_state is None or season is None:
                new_df.loc[t, metric] = scores[i]
            else:
                new_df.loc[(leaf_state, season, t), metric] = scores[i]
        return new_df

    def _calc_single_metric(self, metric: str) -> list:
        """
        Calculates scores for each target using the metric acronym.

        Parameters:
            metric (str): Metric acronym to calculate.e.g., "r2", "mse", "rmse", etc.

        Returns:
            list: List of scores for each target.
        """

        def relative_root_mean_squared_error(y_test, y_pred):
            rmse = root_mean_squared_error(y_test, y_pred)
            baseline = np.sqrt(np.square(sum(y_test)) / len(y_test))
            return rmse / baseline

        def weighted_root_mean_squared_error(y_test, y_pred, weight_index):
            mse = mean_squared_error(y_test, y_pred)
            if self.weights is None:
                raise ValueError(
                    "Weights are not set, yet wRMSE is either elected or calculating by default. Set weights."
                )
            weighted_mse = mse * self.weights[weight_index]
            return np.sqrt(weighted_mse)

        calc_func = None

        if metric == "r2":
            calc_func = r2_score
        elif metric == "mse":
            calc_func = mean_squared_error
        elif metric == "rmse":
            calc_func = root_mean_squared_error
        elif metric == "rrmse":
            calc_func = relative_root_mean_squared_error
        elif metric == "mae":
            calc_func = mean_absolute_error
        elif metric == "mape":
            calc_func = mean_absolute_percentage_error
        elif metric == "wrmse":
            calc_func = weighted_root_mean_squared_error

        scores = []
        for i in range(self.y_test.shape[1]):
            yt = self.y_test.iloc[:, i]
            yp = self.y_pred.iloc[:, i]
            nan_mask = np.isnan(
                yt
            )  # Creates a mask to remove NaN values consistently for any metric
            if metric == "wrmse":
                score = calc_func(yt[~nan_mask], yp[~nan_mask], i)
            else:
                score = calc_func(yt[~nan_mask], yp[~nan_mask])
            scores.append(score)

        return scores

    def aggregate(self, levels: list = None) -> pd.DataFrame:
        """
        Aggregates the results dataframe by the specified levels. If no levels
        are provided, it returns the mean of all results.

        Parameters:
            levels (list): List of levels to aggregate by. If None, aggregates all results.

        Returns:
            pd.DataFrame: Aggregated results.
        """
        if levels is None:
            return pd.DataFrame(
                self.results.reset_index(drop=True).mean(),
                columns=["score"],
                dtype="float",
            )
        if all(level in self.results.index.names for level in levels):
            return self.results.groupby(level=levels).mean()
        else:
            print(
                f"Values given for indices to aggregate by are invalid. Provided: {levels}, Available: {self.results.index.names}"
            )

    def include_element_units(self):
        """
        Adds a column of units to the results dataframe based on the target element.
        The units are defined in the ELEMENT_UNITS dictionary.

        Parameters:
            None

        Returns:
            None
        """
        if "target" not in self.results.index.names:
            raise ValueError("Cannot add units column. There is no target column.")
        elements = self.results.index.get_level_values("target")
        self.results["units"] = [ELEMENT_UNITS.get(element) for element in elements]
        self.results = self.results.set_index("units", append=True)
