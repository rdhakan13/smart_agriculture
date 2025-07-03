import numpy as np
import pandas as pd
from scipy.stats import norm, rankdata
from spectroscopy.src.common.constants import ELEMENT_COLUMNS
from sklearn.preprocessing import MinMaxScaler, StandardScaler, QuantileTransformer


class TargetScaler:
    """
    Class for scaling target columns in a DataFrame using various methods.
    The class supports Min-Max scaling, Standard scaling, Log transformation,
    Quantile transformation, Gaussian Copula transformation, and Rank + Min-Max scaling.
    """

    def __init__(self, scaling_methods: dict, target_columns: list = ELEMENT_COLUMNS):
        """
        Initialize the scaler.

        Parameters:
            scaling_methods (dict): Dictionary mapping target columns to scaling methods.
            target_columns (list): List of target columns to scale. Default is ELEMENT_COLUMNS.
        """
        self.target_columns = target_columns
        self.scaling_methods = scaling_methods
        self.scalers = {}  # Dictionary to hold scalers for each target column
        self.minmax_scalers = {}  # Store Min-Max scalers for final step

    def fit(self, df: pd.DataFrame):
        """
        Fit the scalers using the training dataset.

        Parameters:
            df (pd.DataFrame): DataFrame containing target columns.

        Returns:
            None
        """
        for col in self.target_columns:
            if col not in df.columns:
                continue  # Skip missing columns
            method = self.scaling_methods.get(col, None)

            if method == "minmax":
                scaler = MinMaxScaler()
            elif method == "standard":
                scaler = StandardScaler()
            elif method == "log":
                scaler = "log"  # Log transformation doesn't require fitting
            elif method == "quantile":
                scaler = QuantileTransformer(
                    output_distribution="normal", n_quantiles=min(len(df), 1000)
                )
            elif method == "copula":
                scaler = None  # Gaussian Copula doesn't require fitting
            elif method == "rank_minmax":
                scaler = MinMaxScaler()
            elif method is None:
                scaler = None
            else:
                raise ValueError(f"Unknown scaling method for {col}: {method}")

            # Fit scaler if required
            if method in ["minmax", "standard", "quantile", "rank_minmax"]:
                scaler.fit(df[[col]])  # Keep as DataFrame to preserve feature names

            self.scalers[col] = scaler

            # Fit Min-Max scaler for final step
            minmax_scaler = MinMaxScaler()
            transformed_col = self._apply_transformation(
                df[[col]], scaler, method
            )  # Keep as DataFrame
            minmax_scaler.fit(transformed_col)
            self.minmax_scalers[col] = minmax_scaler

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply scaling to the target columns.

        Parameters:
            df (pd.DataFrame): DataFrame containing target columns.

        Returns:
            pd.DataFrame: DataFrame with scaled target columns.
        """
        df_scaled = df.copy()
        for col in df.columns:  # Iterate only over existing columns
            scaler = self.scalers.get(col, None)
            minmax_scaler = self.minmax_scalers.get(col, None)
            method = self.scaling_methods.get(col, None)

            if scaler is None or minmax_scaler is None:
                continue  # Skip missing columns

            transformed = self._apply_transformation(
                df[[col]], scaler, method
            )  # Keep as DataFrame
            df_scaled[col] = minmax_scaler.transform(transformed)  # Keep as DataFrame

        return df_scaled

    def inverse_transform(self, df_scaled: pd.DataFrame) -> pd.DataFrame:
        """
        Reverse the scaling transformation.

        Parameters:
            df_scaled (pd.DataFrame): DataFrame containing scaled target columns.

        Returns:
            pd.DataFrame: DataFrame with original target columns.
        """
        df_original = df_scaled.copy()
        for col in df_scaled.columns:  # Iterate only over existing columns
            scaler = self.scalers.get(col, None)
            minmax_scaler = self.minmax_scalers.get(col, None)
            method = self.scaling_methods.get(col, None)

            if scaler is None or minmax_scaler is None:
                continue  # Skip missing columns

            # Reverse Min-Max Scaling
            unscaled = minmax_scaler.inverse_transform(
                df_scaled[[col]]
            )  # Keep as DataFrame

            # Reverse primary transformation
            df_original[col] = self._reverse_transformation(
                pd.DataFrame(unscaled, columns=[col]), scaler, method
            )

        return df_original

    @staticmethod
    def _apply_transformation(
        col_values: pd.DataFrame, scaler, method: str
    ) -> pd.DataFrame:
        """
        Apply the selected transformation.

        Parameters:
            col_values (pd.DataFrame): Column values to transform.
            scaler: The scaler to apply.
            method: Transformation method (e.g., "log", "minmax", "standard", "quantile", "copula").

        Returns:
            pd.DataFrame: Transformed values as a DataFrame.
        """
        if method == "log":
            transformed = np.log1p(col_values)  # Log transformation
        elif method in ["minmax", "standard", "quantile"]:
            transformed = pd.DataFrame(
                scaler.transform(col_values), columns=col_values.columns
            )
        elif method == "copula":
            ranked = rankdata(col_values) / (len(col_values) + 1)  # Avoid exact 0 or 1
            ranked = np.clip(ranked, 1e-6, 1 - 1e-6)
            transformed = pd.DataFrame(
                norm.ppf(ranked), columns=col_values.columns
            )  # Gaussian Copula transformation
        elif method == "rank_minmax":
            ranked = rankdata(col_values) / len(col_values)
            transformed = pd.DataFrame(
                scaler.transform(pd.DataFrame(ranked, columns=col_values.columns)),
                columns=col_values.columns,
            )  # Rank + MinMax Scaling
        elif method is None:
            transformed = col_values
        else:
            raise ValueError(f"Unknown scaling method: {method}")

        return transformed

    @staticmethod
    def _reverse_transformation(
        transformed_values: pd.DataFrame, scaler, method: str
    ) -> pd.DataFrame:
        """
        Reverse the selected transformation.

        Parameters:
            transformed_values (pd.DataFrame): Transformed values to reverse.
            scaler: The scaler to apply.
            method: Transformation method (e.g., "log", "minmax", "standard", "quantile", "copula").

        Returns:
            pd.DataFrame: Original values as a DataFrame.
        """
        if method == "log":
            return np.expm1(transformed_values)
        elif method in ["minmax", "standard", "quantile"]:
            return pd.DataFrame(
                scaler.inverse_transform(transformed_values),
                columns=transformed_values.columns,
            )
        elif method == "copula":
            norm_cdf = norm.cdf(transformed_values)
            return pd.DataFrame(
                np.interp(
                    norm_cdf,
                    (norm_cdf.min(), norm_cdf.max()),
                    (transformed_values.min(), transformed_values.max()),
                ),
                columns=transformed_values.columns,
            )
        elif method == "rank_minmax":
            ranked_reversed = scaler.inverse_transform(transformed_values)
            return pd.DataFrame(
                np.interp(
                    ranked_reversed,
                    (0, 1),
                    (transformed_values.min(), transformed_values.max()),
                ),
                columns=transformed_values.columns,
            )
        else:
            raise ValueError(f"Unknown scaling method: {method}")
