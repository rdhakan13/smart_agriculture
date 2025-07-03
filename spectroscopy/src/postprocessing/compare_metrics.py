import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from spectroscopy import Metrics


class CompareMetrics:
    """
    Class for comparing two sets of metrics results.
    This class accepts two dataframes of metrics results and calculates the 
    difference, percentage, or ratio between them.
    It also provides a method to visualize the comparison results using a heatmap.
    """

    def __init__(
        self,
        a_results: pd.DataFrame,
        b_results: pd.DataFrame,
        calculation: str = "percentage",
    ):
        """
        Class accepts two dataframes and a calc string which determines the calculation
        method for the comparison. Both dataframes must be limited to simple single-index
        tables with the targets as the row-index.

        Parameters:
            a_results (pd.DataFrame): First dataframe of metrics results.
            b_results (pd.DataFrame): Second dataframe of metrics results.
            calculation (str): Calculation method ('difference', 'percentage', 'ratio').

        Returns:
            None
        """
        if isinstance(a_results.index, pd.MultiIndex) or isinstance(
            b_results.index, pd.MultiIndex
        ):
            print(
                "DataFrame has a multilevel index. Function requires that both dataframes have a single level row index of element names."
            )
            print("No comparison dataframe has been created.")
            return

        self.calc = calculation
        self.a = a_results
        self.b = b_results
        self.compare_results()
        self.visualise()

    def compare_results(self, calculation=None):
        """
        This produces the calculated results based on the given calculation method.
        This can be rerun with a different method, once the class is loaded.

        Parameters:
            calculation (str): Calculation method ('difference', 'percentage', 'ratio').

        Returns:
            None
        """
        if calculation is None:
            calculation = self.calc

        targets = self.a.index.get_level_values("target")
        metrics = self.a.columns

        if self.a.shape != self.b.shape:
            print(
                f"Shape of results are not equal: a_result={self.a.shape}, b_result={self.b.shape}"
            )
            print("Refactoring so that they are equal...")

            if self.a.shape[0] != self.b.shape[0]:
                targets = self._compare_targets(self.a, self.b)

            if self.a.shape[1] != self.b.shape[1]:
                metrics = self._compare_metric_cols(self.a, self.b)

        self.comparison = self._calculate_difference(
            self.a.loc[targets, metrics], self.b.loc[targets, metrics], calculation
        )

    def _compare_targets(self) -> list:
        """
        Function to compare the targets of the two dataframes.

        Parameters:
            None

        Returns:
            list: List of common targets between the two dataframes.
        """
        a_targets = set(self.a.index.get_level_targets("target").to_list())
        b_targets = set(self.b.index.get_level_targets("target").to_list())
        new_targets = a_targets.intersection(b_targets)

        print("Results show following targets exist in one set and not other:")
        print(f"\ta_targets={a_targets - b_targets}, b_targets={b_targets - a_targets}")
        print(
            f"Only comparing metrics for {new_targets} targets, excluding {new_targets.difference(a_targets + b_targets)}"
        )

        return list(new_targets)

    def _compare_metric_cols(self) -> list:
        """
        Function to compare the columns of the two dataframes.

        Parameters:
            None

        Returns:
            list: List of common columns between the two dataframes.
        """
        a_cols = set(self.a.columns.to_list())
        b_cols = set(self.b.columns.to_list())
        new_cols = a_cols.intersection(b_cols)

        print(
            f"Results show different number of metrics. Common metrics between both result sets are: {new_cols}"
        )

        return list(new_cols)

    @staticmethod
    def _calculate_difference(
        a: pd.DataFrame, b: pd.DataFrame, calc: str
    ) -> pd.DataFrame:
        """
        Function to calculate the difference between two dataframes based on the
        given calculation method.

        Parameters:
            a (pd.DataFrame): First dataframe.
            b (pd.DataFrame): Second dataframe.
            calc (str): Calculation method ('difference', 'percentage', 'ratio').

        Returns:
            pd.DataFrame: DataFrame containing the calculated differences.
        """
        try:
            if calc == "difference":
                return b - a
            if calc == "percentage":
                return (b - a) / a * 100
            if calc == "ratio":
                return b / a
        except Exception as e:
            raise ValueError(
                f"Failed to calculate the difference between two dataframes, with the following error:"
                f"\n\t{e}"
            )

    def visualise(self):
        """
        Function to help visualise results

        Parameters:
            None

        Returns:
            None
        """
        r = self.comparison.copy()
        r_error_cols = [col for col in r.columns if col != "r2"]
        if (
            self.calc == "percentage"
        ):  # Change signs for errors to positive where there is improvement
            r[r_error_cols] = -r[r_error_cols]
        if self.calc == "ratio":
            r[r_error_cols] = 1 + (1 - r[r_error_cols])
        figure, ax = plt.subplots(figsize=(10, 10))
        sns.set_style("whitegrid")
        hm = sns.heatmap(r, center=0, cmap="vlag", annot=True, ax=ax)
        hm.set_title(
            f"Heatmap of comparison of model (b) to model (a) using calculation '{self.calc}'"
        )
