import logging
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.impute import KNNImputer
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import pairwise_distances_argmin_min
from spectroscopy import LeafSampleReader
from spectroscopy.src.common.constants import ELEMENT_COLUMNS
from spectroscopy.src.preprocessing.data_reducer import DataReducer
from spectroscopy.src.common.utility_functions import get_feature_columns
from sklearn.linear_model import LinearRegression


class DataCleaner:
    """
    Class for cleaning and preprocessing data, including handling missing values,
    outlier removal, and data type enforcement.
    """

    @staticmethod
    def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
        """
        Clean the dataframe by enforcing data types, dropping null values, and imputing missing data.

        Parameters:
            df: DataFrame to be cleaned.

        Returns:
            df: Cleaned DataFrame.
        """
        df = DataCleaner.enforce_data_types(df)
        df = DataCleaner.drop_null_data(df)
        df = DataCleaner.impute_data(df)
        return df

    @staticmethod
    def enforce_data_types(df: pd.DataFrame) -> pd.DataFrame:
        """
        Enforce expected datatypes of the dataframe, converting non-numeric values to NaN.

        Parameters:
            df: DataFrame to be cleaned.

        Returns:
            df: DataFrame with enforced data types.
        """
        df = df.copy()
        element_columns = [col for col in df.columns if col in ELEMENT_COLUMNS]
        df[element_columns] = df[element_columns].apply(pd.to_numeric, errors="coerce")
        feature_columns = get_feature_columns(df)
        df[feature_columns] = df[feature_columns].apply(pd.to_numeric, errors="coerce")
        return df

    @staticmethod
    def drop_null_data(
        df: pd.DataFrame,
        row_threshold: float = 0.5,
        target_col_threshold: float = 0.5,
        feature_col_threshold: float = 0.5,
        reset_index: bool = True,
    ) -> pd.DataFrame:
        """
        Drop rows and columns with too many missing values and return a new DataFrame.

        Parameters:
            df: DataFrame to be cleaned.
            row_threshold: Maximum percentage of null values allowed in a row.
            target_col_threshold: Maximum percentage of null values allowed in target columns.
            feature_col_threshold: Maximum percentage of null values allowed in feature columns.
            reset_index: Whether to reset the index of the DataFrame after dropping rows/columns.

        Returns:
            df: DataFrame with rows and columns dropped based on null value thresholds.
        """
        df = df.copy()
        df = DataCleaner._drop_rows_with_too_many_nulls(df, row_threshold)
        df = DataCleaner._drop_target_columns_with_too_many_nulls(
            df, target_col_threshold
        )
        df = DataCleaner._drop_feature_columns_with_too_many_nulls(
            df, feature_col_threshold
        )
        # Avoids fragmentation
        if reset_index:
            df = df.reset_index(drop=True)
        return df

    @staticmethod
    def _drop_rows_with_too_many_nulls(
        df: pd.DataFrame, max_null_pct: float
    ) -> pd.DataFrame:
        """
        Drop rows where the percentage of null values is greater than or equal to max_null_pct.

        Parameters:
            df: DataFrame to be cleaned.
            max_null_pct: Maximum percentage of null values allowed in a row.

        Returns:
            df: DataFrame with rows dropped based on null value threshold.
        """
        rows_to_drop = df.isnull().mean(axis=1) >= max_null_pct
        indices_to_drop = df.index[rows_to_drop]
        df = df.drop(index=indices_to_drop)

        if rows_to_drop.any():
            logging.info(
                f"Rows {list(indices_to_drop)} have more than {(max_null_pct * 100):.0f}% Null and were dropped"
            )
        return df

    @staticmethod
    def _drop_columns_with_too_many_nulls(
        df: pd.DataFrame, columns_to_review: list[str], max_null_pct: float
    ) -> pd.DataFrame:
        """
        Drop columns where the percentage of null values is greater than or equal to max_null_pct.

        Parameters:
            df: DataFrame to be cleaned.
            columns_to_review: List of columns to check for null values.
            max_null_pct: Maximum percentage of null values allowed in a column.

        Returns:
            df: DataFrame with columns dropped based on null value threshold.
        """
        cols_to_drop = [
            col
            for col in columns_to_review
            if col in df.columns and df[col].isnull().mean() >= max_null_pct
        ]
        pct_nans = df[cols_to_drop].isnull().mean()
        df = df.drop(
            columns=cols_to_drop
        ).copy()  # copy dataframe to avoid fragmentation
        if cols_to_drop:
            logging.info(
                f"Columns {cols_to_drop} have more than {(max_null_pct * 100):.0f}% Null and were dropped, "
                f"with {[round(x * 100, 1) for x in pct_nans]}% respectively."
            )
        return df

    @staticmethod
    def _drop_target_columns_with_too_many_nulls(
        df: pd.DataFrame, max_null_pct: float
    ) -> pd.DataFrame:
        """
        Drop target columns (elements) where the percentage of null values is greater than or equal to max_null_pct.

        Parameters:
            df: DataFrame to be cleaned.
            max_null_pct: Maximum percentage of null values allowed in target columns.

        Returns:
            df: DataFrame with target columns dropped based on null value threshold.
        """
        return DataCleaner._drop_columns_with_too_many_nulls(
            df, ELEMENT_COLUMNS, max_null_pct
        )

    @staticmethod
    def _drop_feature_columns_with_too_many_nulls(
        df: pd.DataFrame, max_null_pct: float
    ) -> pd.DataFrame:
        """
        Drop feature columns (wavelengths) where the percentage of null values is greater than or equal to max_null_pct.

        Parameters:
            df: DataFrame to be cleaned.
            max_null_pct: Maximum percentage of null values allowed in feature columns.

        Returns:
            df: DataFrame with feature columns dropped based on null value threshold.
        """
        feature_columns = get_feature_columns(df)
        return DataCleaner._drop_columns_with_too_many_nulls(
            df, feature_columns, max_null_pct
        )

    @staticmethod
    def impute_data(
        df: pd.DataFrame,
        target_method: str = "knn",
        knn_neighbors: int = 3,
        feature_method: str = "neighbour_avg",
    ) -> pd.DataFrame:
        """
        Impute missing values in both features and target columns.

        Parameters:
            df: DataFrame to be cleaned.
            target_method: Method for imputing target columns ('pearson', 'regression', 'knn').
            knn_neighbors: Number of neighbors for KNN imputation.
            feature_method: Method for imputing feature columns ('neighbour_avg', 'polynomial_fitting').

        Returns:
            df: DataFrame with missing values imputed.
        """
        df = df.copy()
        df = DataCleaner._impute_target_data(df, target_method, knn_neighbors)
        df = DataCleaner._impute_feature_data(df, feature_method)
        return df

    @staticmethod
    def _impute_feature_data(df: pd.DataFrame, method="neighbour_avg") -> pd.DataFrame:
        """
        Impute missing values in feature columns using selected method.

        Parameters:
            df: DataFrame to be cleaned.
            method: Method for imputing feature columns ('neighbour_avg', 'polynomial_fitting').

        Returns:
            df: DataFrame with missing values in feature columns imputed.
        """
        if method is None:
            return df
        if method == "neighbour_avg":
            return DataCleaner._impute_feature_with_average_of_neighbours(df)
        elif method == "polynomial_fitting":
            return DataCleaner._impute_features_with_polynomial_fitting(df)
        else:
            raise ValueError(
                "Invalid method. Choose 'neighbour_avg' or 'polynomial_fitting'."
            )

    @staticmethod
    def _impute_features_with_polynomial_fitting(df: pd.DataFrame) -> pd.DataFrame:
        pass

    @staticmethod
    def _impute_feature_with_average_of_neighbours(df: pd.DataFrame) -> pd.DataFrame:
        """
        Imputes missing values in feature columns using the average of neighboring wavelengths.

        Parameters:
            df: DataFrame to be cleaned.

        Returns:
            df: DataFrame with missing values in feature columns imputed.
        """
        feature_columns = get_feature_columns(df)
        DataCleaner._log_nans(df[feature_columns], "Feature")
        df_prev = df[feature_columns].shift(axis=1)
        df_next = df[feature_columns].shift(-1, axis=1)
        df_avg = (df_prev + df_next) / 2
        df[feature_columns] = df[feature_columns].fillna(df_avg)
        df = df.copy()
        df[feature_columns] = df[feature_columns].fillna(df_prev).fillna(df_next)
        return df.copy()

    @staticmethod
    def _impute_target_data(
        df: pd.DataFrame, method=None, knn_neighbors: int = 3
    ) -> pd.DataFrame:
        """
        Impute missing values in element (target) columns using the selected method.

        Parameters:
            df: DataFrame to be cleaned.
            method: Method for imputing target columns ('pearson', 'regression', 'knn').
            knn_neighbors: Number of neighbors for KNN imputation.

        Returns:
            df: DataFrame with missing values in target columns imputed.
        """
        if method is None:
            return df
        if method == "pearson":
            return DataCleaner._impute_targets_with_pearson(df)
        elif method == "regression":
            return DataCleaner._impute_targets_with_regression(df)
        elif method == "knn":
            return DataCleaner._impute_targets_with_knn(df, knn_neighbors)
        else:
            raise ValueError(
                "Invalid method. Choose 'pearson', 'regression', or 'knn'."
            )

    @staticmethod
    def _impute_targets_with_knn(
        df: pd.DataFrame, knn_neighbors: int = 3
    ) -> pd.DataFrame:
        """
        Impute missing values in target columns using KNN imputation.

        Parameters:
            df: DataFrame to be cleaned.
            knn_neighbors: Number of neighbors for KNN imputation.

        Returns:
            df: DataFrame with missing values in target columns imputed.
        """
        element_columns = [col for col in df.columns if col in ELEMENT_COLUMNS]
        DataCleaner._log_nans(df[element_columns], "Target")
        knn_imputer = KNNImputer(n_neighbors=knn_neighbors)
        df[element_columns] = knn_imputer.fit_transform(df[element_columns])
        return df

    @staticmethod
    def _impute_targets_with_pearson(df: pd.DataFrame):
        """
        Impute missing values in target columns using Pearson correlation.

        Parameters:
            df: DataFrame to be cleaned.

        Returns:
            df: DataFrame with missing values in target columns imputed.
        """
        element_columns = [col for col in df.columns if col in ELEMENT_COLUMNS]
        DataCleaner._log_nans(df[element_columns], "Target")
        df[element_columns] = df[element_columns].apply(pd.to_numeric, errors="coerce")
        correlation_matrix = df[element_columns].corr(method="pearson")

        for element in element_columns:
            missing_indices = df[df[element].isnull()].index

            if missing_indices.empty:
                continue

            correlated_elements = correlation_matrix[element].drop(element)
            correlated_elements = correlated_elements[
                abs(correlated_elements) > 0.5
            ].index.tolist()

            if not correlated_elements:
                continue

            for idx in missing_indices:
                row = df.loc[idx]
                weighted_sum = 0.0
                total_weight = 0.0

                for other in correlated_elements:
                    if pd.notnull(row[other]):
                        weight = abs(correlation_matrix.loc[element, other])
                        weighted_sum += row[other] * weight
                        total_weight += weight

                if total_weight > 0:
                    df.at[idx, element] = weighted_sum / total_weight
        return df

    @staticmethod
    def _impute_targets_with_regression(df: pd.DataFrame):
        """
        Impute missing values in target columns using regression.

        Parameters:
            df: DataFrame to be cleaned.

        Returns:
            df: DataFrame with missing values in target columns imputed.
        """
        element_columns = [col for col in df.columns if col in ELEMENT_COLUMNS]
        DataCleaner._log_nans(df[element_columns], "Target")
        df[element_columns] = df[element_columns].apply(pd.to_numeric, errors="coerce")
        correlation_matrix = df[element_columns].corr(method="pearson")

        for element in element_columns:
            missing_indices = df[df[element].isnull()].index
            if missing_indices.empty:
                continue

            correlated_elements = correlation_matrix[element].drop(element)
            correlated_elements = correlated_elements[
                abs(correlated_elements) > 0.5
            ].index.tolist()

            if not correlated_elements:
                continue

            train_data = df.dropna(subset=[element] + correlated_elements)
            if train_data.shape[0] < 5:
                continue

            X_train = train_data[correlated_elements]
            y_train = train_data[element]

            model = LinearRegression()
            model.fit(X_train, y_train)

            X_test = df.loc[missing_indices, correlated_elements]
            X_test = X_test.dropna(axis=0)

            if X_test.empty:
                continue

            predictions = model.predict(X_test)
            df.loc[X_test.index, element] = predictions

        return df

    @staticmethod
    def _log_nans(df: pd.DataFrame, df_name: str):
        """
        Log the number of missing values in the DataFrame.

        Parameters:
            df: DataFrame to be checked for missing values.
            df_name: Name of the DataFrame for logging purposes.

        Returns:
            None
        """
        nan_count = df.isnull().sum()
        total_nans = nan_count.sum()
        count_by_col = list(nan_count[nan_count > 0].items())
        nan_percent = (total_nans / df.size) * 100
        if total_nans > 0:
            logging.info(
                f"{df_name} data is missing {nan_count.sum()} values ({nan_percent:.3f}%): "
                f"{count_by_col if len(count_by_col) < 10 else '(too many columns to list)'}."
            )

    @staticmethod
    def remove_outliers(
        df: pd.DataFrame,
        method="both",
        n_estimators: int = 5000,
        random_state: int = 42,
        feature_outlier_threshold: int = 95,
    ) -> pd.DataFrame:
        """
        Remove outliers from the DataFrame using the specified method.

        Parameters:
            df: DataFrame to be cleaned.
            method: Method for removing outliers ('targets', 'features', 'both', None).
            n_estimators: Number of estimators for Isolation Forest.
            random_state: Random state for reproducibility.
            feature_outlier_threshold: Percentage threshold for identifying outliers in features.

        Returns:
            cleaned_df: DataFrame with outliers removed.
            outlier_indices: List of indices of the removed outliers.
        """
        df = df.reset_index(drop=True)
        if method == "targets":
            return DataCleaner._remove_outliers_target(df, n_estimators, random_state)
        elif method == "features":
            return DataCleaner._remove_outliers_features(
                df, outlier_threshold=feature_outlier_threshold
            )
        elif method == "both":
            _, targets_indices = DataCleaner._remove_outliers_target(
                df, n_estimators, random_state
            )
            _, features_indices = DataCleaner._remove_outliers_features(
                df, outlier_threshold=feature_outlier_threshold
            )
            set1, set2 = set(targets_indices), set(features_indices)
            outlier_indices = list(set1.symmetric_difference(set2))
            cleaned_df = df.drop(index=outlier_indices, axis=0).reset_index(drop=True)
            return cleaned_df, outlier_indices
        elif method is None:
            return df, []
        else:
            raise ValueError("Invalid method. Choose 'targets', 'features', or 'both'.")

    @staticmethod
    def _remove_outliers_target(
        df: pd.DataFrame, n_estimators: int = 5000, random_state=42
    ) -> pd.DataFrame:
        """
        Remove outliers from target columns using Isolation Forest.

        Parameters:
            df: DataFrame to be cleaned.
            n_estimators: Number of estimators for Isolation Forest.
            random_state: Random state for reproducibility.

        Returns:
            cleaned_df: DataFrame with outliers removed.
            outlier_indices: List of indices of the removed outliers.
        """
        scaler = MinMaxScaler()
        targets_df = LeafSampleReader.extract_targets(df)
        scaled_df = scaler.fit_transform(targets_df)
        iso_forest = IsolationForest(
            n_estimators=n_estimators, random_state=random_state
        )
        outliers = iso_forest.fit_predict(scaled_df)
        outlier_indices = df.index[outliers == -1]
        cleaned_df = df.drop(index=outlier_indices, axis=0).reset_index(drop=True)
        return cleaned_df, list(outlier_indices)

    @staticmethod
    def _remove_outliers_features(
        df: pd.DataFrame, outlier_threshold: int = 95
    ) -> pd.DataFrame:
        """
        Remove outliers from feature columns using KMeans clustering.

        Parameters:
            df: DataFrame to be cleaned.
            outlier_threshold: Percentage threshold for identifying outliers.

        Returns:
            cleaned_df: DataFrame with outliers removed.
            outlier_indices: List of indices of the removed outliers.
        """
        season_cluster_means = []
        outlier_indices = []
        dr = DataReducer(method="pca")
        reduced_data = dr.reduce_data(df)
        for season in df["season"].unique():
            season_df = reduced_data[["PC1", "PC2"]][reduced_data["season"] == season]
            kmeans = KMeans(n_clusters=1, random_state=42)
            kmeans.fit(season_df)
            season_cluster_means.append(kmeans.cluster_centers_)
            distances = pairwise_distances_argmin_min(
                reduced_data[reduced_data["season"] == season][["PC1", "PC2"]],
                kmeans.cluster_centers_,
            )[1]
            threshold = np.percentile(distances, outlier_threshold)
            outlier_indices.extend(
                list(
                    reduced_data[reduced_data["season"] == season][["PC1", "PC2"]][
                        distances > threshold
                    ].index
                )
            )
        cleaned_df = df.drop(index=outlier_indices, axis=0).reset_index(drop=True)
        return cleaned_df, outlier_indices
