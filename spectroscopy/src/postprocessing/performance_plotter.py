import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt


class PerformancePlotter:
    """
    Class for plotting performance metrics of machine learning models.
    This class provides methods to visualize predictions, residuals, R² scores,
    and RMSE values.
    It also includes a method for inspecting the data in a DataFrame.
    """

    @staticmethod
    def data_inspection(df):
        """
        Inspect the given DataFrame by printing key information and summaries.

        Parameters:
            df (pd.DataFrame): The DataFrame to be inspected.

        Returns:
            None
        """
        print("First 5 rows of the DataFrame:")
        print(df.head())
        print("\nDataFrame Information:")
        df.info()
        print("\nDescriptive Statistics:")
        print(df.describe())

    @staticmethod
    def plot_predictions(
        y_true, y_pred, title: str = "Model Predictions", legend: bool = False
    ):
        """
        Plot true vs predicted values.

        Parameters:
            y_true (pd.DataFrame or np.ndarray): True values.
            y_pred (pd.DataFrame or np.ndarray): Predicted values.
            title (str): Title of the plot.
            legend (bool): Whether to include a legend.

        Returns:
            None
        """
        plt.figure(figsize=(9, 6))
        if legend:
            results_df = PerformancePlotter._create_long_df(y_true, y_pred)
            sns.scatterplot(
                data=results_df,
                x="Value_x",
                y="Value_y",
                hue="Nutrient",
                palette="tab20",
            )
            plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        else:
            plt.scatter(y_true, y_pred, alpha=0.7)
        plt.plot(
            [y_true.min(), y_true.max()],
            [y_true.min(), y_true.max()],
            color="red",
            linestyle="--",
        )
        plt.xlabel("True Values")
        plt.ylabel("Predicted Values")
        plt.title(title)
        plt.grid(True)
        plt.show()

    @staticmethod
    def plot_residuals(
        y_true, y_pred, title: str = "Residual Plot", legend: bool = False
    ):
        """
        Plot residuals (true - predicted).

        Parameters:
            y_true (pd.DataFrame or np.ndarray): True values.
            y_pred (pd.DataFrame or np.ndarray): Predicted values.
            title (str): Title of the plot.
            legend (bool): Whether to include a legend.

        Returns:
            None
        """
        plt.figure(figsize=(9, 6))
        if legend:
            results_df = PerformancePlotter._create_long_df(y_true, (y_pred - y_true))
            sns.scatterplot(
                data=results_df,
                x="Value_x",
                y="Value_y",
                hue="Nutrient",
                palette="tab20",
            )
            plt.legend(bbox_to_anchor=(1.02, 1), loc="upper left")
        else:
            residuals = y_true - y_pred
            plt.scatter(y_true, residuals, alpha=0.7)
        plt.axhline(y=0, color="red", linestyle="--")
        plt.xlabel("True Values")
        plt.ylabel("Residuals")
        plt.title(title)
        plt.grid(True)
        plt.tight_layout()
        plt.show()

    @staticmethod
    def _create_long_df(x, y) -> pd.DataFrame:
        """
        Create a long DataFrame for plotting.

        Parameters:
            x (pd.DataFrame or np.ndarray): First DataFrame or array.
            y (pd.DataFrame or np.ndarray): Second DataFrame or array.

        Returns:
            pd.DataFrame: Long DataFrame with melted values.
        """
        params = {"value_name": "Value", "var_name": "Nutrient"}
        x_long = x.melt(**params)
        new_index = pd.MultiIndex.from_arrays([x_long.index, x_long["Nutrient"]])
        x_long = x_long.set_index(new_index, drop=True)
        if isinstance(y, np.ndarray):
            y = pd.DataFrame(y)
        y_long = y.melt(**params).set_index(new_index, drop=True)
        return x_long.join(y_long, how="left", lsuffix="_x", rsuffix="_y")

    @staticmethod
    def plot_r2(r_squared, title: str = "R² Score"):
        """
        Plot the R² score.

        Parameters:
            r_squared (float): The R² score value.
            title (str): Title of the plot.

        Returns:
            None
        """
        plt.figure(figsize=(6, 4))
        plt.bar(["R² Score"], [r_squared], color="skyblue")
        plt.ylim(0, 1)
        plt.ylabel("Score")
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()

    @staticmethod
    def plot_rmse(mse, title: str = "RMSE"):
        """
        Plot the RMSE value.

        Parameters:
            mse (float): The mean squared error value.
            title (str): Title of the plot.

        Returns:
            None
        """
        rmse = mse**0.5
        plt.figure(figsize=(6, 4))
        plt.bar(["RMSE"], [rmse], color="salmon")
        plt.ylabel("Value")
        plt.title(title)
        plt.grid(True, linestyle="--", alpha=0.6)
        plt.show()
