import os
import pywt
import pentapy
import logging
import pandas as pd
import numpy as np
from datetime import datetime as dt
import matplotlib.pyplot as plt
from sklearn.decomposition import FastICA
from spectroscopy.src.common.utility_functions import (
    get_feature_columns,
    get_working_directory,
)


class BaselineCorrector:
    """
    A static class for applying various baseline correction techniques to spectral data.

    Supports multiple methods: SNV, Polynomial, Wavelet, Iterative Mean, ICA, ALS, Detrending.
    """

    @staticmethod
    def correct_dataframe(
        df: pd.DataFrame,
        method: str = "snv",
        mean_baseline: bool = True,
        rescale: bool = True,
        **kwargs,
    ):
        """
        Apply baseline correction and return both the corrected DataFrame and 
        the estimated baseline.

        Parameters:
            df (pd.DataFrame): The input DataFrame with spectral data.
            method (str): The baseline correction method.
            mean_baseline (bool): Whether to calculate a single baseline base on mean of samples or one for each sample
            rescale (bool): Whether to rescale corrected values to [0,1].
            **kwargs: Additional parameters for specific methods.

        Returns:
            pd.DataFrame: The baseline-corrected DataFrame.
            pd.DataFrame: The estimated baseline (matches shape of input).
        """
        start = dt.now()
        corrected_df = df.copy()
        feature_columns = get_feature_columns(df)

        baseline = BaselineCorrector._apply_baseline_correction(
            df[feature_columns], method, mean_based=mean_baseline, **kwargs
        )
        if method is not None:
            corrected_df[feature_columns] -= baseline

        if rescale:
            corrected_df = BaselineCorrector.rescale_corrected_data(corrected_df)

        logging.info(
            f"Baseline correction for {method} completed in {dt.now() - start}"
        )

        return corrected_df, baseline

    @staticmethod
    def rescale_corrected_data(df_corrected: pd.DataFrame) -> pd.DataFrame:
        """
        Rescales corrected spectral data to be in the range [0,1] using Min-Max Scaling.

        Parameters:
            df_corrected (pd.DataFrame): The baseline-corrected spectral data.

        Returns:
            pd.DataFrame: Rescaled corrected data.
        """
        feature_columns = get_feature_columns(df_corrected)

        # Compute global min and max across all feature columns
        X_min = df_corrected[feature_columns].min().min()
        X_max = df_corrected[feature_columns].max().max()

        # Apply Min-Max Scaling
        df_rescaled = df_corrected.copy()
        df_rescaled[feature_columns] = (df_corrected[feature_columns] - X_min) / (
            X_max - X_min
        )

        return df_rescaled

    @staticmethod
    def _apply_baseline_correction(
        X: pd.DataFrame, method: str, mean_based=True, **kwargs
    ) -> pd.DataFrame:
        """
        Apply the selected baseline correction method and return the estimated baseline.

        Parameters:
            X (pd.DataFrame): Feature columns (wavelength data).
            method (str): The baseline correction method.
            mean_based (bool): If True, computes a single baseline for the mean spectrum.
            **kwargs: Additional method parameters.

        Returns:
            pd.DataFrame: The estimated baseline (same shape as input X).
        """
        method_mapping = {
            "polynomial": BaselineCorrector._polynomial_fitting,
            "wavelet": BaselineCorrector._wavelet_transform,
            "iterative_mean": BaselineCorrector._iterative_mean_correction,
            "ica": BaselineCorrector._ica_baseline,
            "als": BaselineCorrector._als_baseline,
        }

        if method is not None and method not in method_mapping:
            raise ValueError(f"Invalid method '{method}'.")
        elif method is None:
            return X
        elif method == "ica" and not mean_based:
            # Special handling: apply ICA to the entire dataset at once
            return BaselineCorrector._ica_baseline(X, **kwargs)
        elif mean_based:
            # Compute baseline on the mean spectrum
            baseline = method_mapping[method](X.mean(axis=0), **kwargs)
            return pd.DataFrame(
                [baseline.values] * X.shape[0], index=X.index, columns=X.columns
            )  # Duplicate for all samples
        else:
            return X.apply(lambda row: method_mapping[method](row, **kwargs), axis=1)

    @staticmethod
    def _polynomial_fitting(X: pd.Series, poly_order=3) -> pd.Series:
        """
        Polynomial baseline correction using a fitted polynomial trend.

        Parameters:
            X (pd.Series): Single spectrum or mean spectrum.
            poly_order (int): Order of the polynomial fit.

        Returns:
            pd.Series: Estimated baseline.
        """
        x_vals = np.arange(len(X))
        poly_coeffs = np.polyfit(x_vals, X.values, poly_order)
        trend = np.polyval(poly_coeffs, x_vals)
        row_baseline = pd.Series(trend, index=X.index)
        return row_baseline

    @staticmethod
    def _wavelet_transform(X: pd.Series, wavelet="db4", level=1) -> pd.Series:
        """
        Wavelet Transform Baseline Correction.
        Uses Discrete Wavelet Transform (DWT) to estimate the baseline.

        Parameters:
            X (pd.Series): Single spectrum or mean spectrum.
            wavelet (str): Wavelet function.
            level (int): Decomposition level.

        Returns:
            pd.Series: Estimated baseline.
        """
        coeffs = pywt.wavedec(X.values, wavelet, mode="smooth", level=level)
        coeffs[-1] = np.zeros_like(coeffs[-1])  # Zero out high-frequency coefficients

        # Reconstruct baseline and ensure correct length
        baseline = pywt.waverec(coeffs, wavelet)[: len(X)]
        return pd.Series(baseline, index=X.index)

    @staticmethod
    def _iterative_mean_correction(X: pd.Series, iterations=5) -> pd.Series:
        """
        Iterative Mean Baseline Correction.

        Parameters:
            X (pd.Series): Single spectrum or mean spectrum.
            iterations (int): Number of iterations.

        Returns:
            pd.Series: Estimated baseline.
        """
        X_corr = X.copy()

        for _ in range(iterations):
            mean_spectrum = X_corr.mean()  # Compute mean value
            X_corr -= mean_spectrum

        return pd.Series(X_corr.mean(), index=X.index)

    @staticmethod
    def _ica_baseline(X: pd.DataFrame, n_components=5) -> pd.Series:
        """
        Independent Component Analysis (ICA) Baseline Correction.

        Parameters:
            X (pd.DataFrame): Feature columns (single or multiple samples).
            n_components (int): Number of ICA components to extract.

        Returns:
            pd.Series: Estimated baseline for a single spectrum or mean spectrum.
        """
        ica = FastICA(n_components=n_components, random_state=42)

        transformed = ica.fit_transform(X.values)  # shape: (n_samples, n_components)
        reconstructed = np.outer(
            transformed[:, 0], ica.mixing_[:, 0]
        )  # shape: (n_samples, n_features)
        baseline = pd.DataFrame(reconstructed, index=X.index, columns=X.columns)

        return baseline

    @staticmethod
    def _als_pentapy(y, lam=1e4, p=0.01, niter=10, eps=1e-10):
        """
        Optimized ALS using Pentapy for efficient pentadiagonal system solving.

        Parameters:
            y (np.array): Input spectrum (1D array)
            lam (float): Regularization parameter (default 1e4)
            p (float): Asymmetry parameter (default 0.01)
            niter (int): Number of iterations (default 10)
            eps (float): Small stabilization factor to prevent singularity (default 1e-10)

        Returns:
            np.array: The estimated baseline.
        """
        L = len(y)

        if L < 3:
            return y  # Avoid issues with very small arrays

        # Construct second-order difference matrix D (L-2 x L)
        D = np.zeros((L - 2, L))
        for i in range(L - 2):
            D[i, i] = 1
            D[i, i + 1] = -2
            D[i, i + 2] = 1

        w = np.ones(L)

        for _ in range(niter):
            W = np.diag(w)  # Convert to diagonal matrix
            A = W + lam * (D.T @ D) + eps * np.eye(L)  # Ensure correct shape
            b = W @ y  # Right-hand side vector

            # Solve using Pentapy
            Z = pentapy.solve(A, b)

            # Update weights
            w = np.clip(
                p * (y > Z) + (1 - p) * (y < Z), 1e-6, 1
            )  # Prevent zero weights

        return Z

    @staticmethod
    def _als_baseline(X: pd.Series, lam=1e4, p=0.01, niter=10) -> pd.Series:
        logging.debug("als running")
        """ 
        Numba-accelerated ALS for a single spectrum. 
        
        Parameters:
            X (pd.Series): Single spectrum.
            lam (float): Regularization parameter.
            p (float): Asymmetry parameter.
            niter (int): Number of iterations.
            
        Returns:
            pd.Series: Estimated baseline.
        """
        return pd.Series(
            BaselineCorrector._als_pentapy(X.values, lam, p, niter), index=X.index
        )

    @staticmethod
    def compare_correction(
        original_df: pd.DataFrame,
        corrected_dfs: dict,
        sample_index=None,
        output_name=None,
        title=None,
    ):
        """
        Plot the original spectrum vs. different baseline correction methods.

        Parameters:
            df_original (pd.DataFrame): The original uncorrected DataFrame.
            corrected_dfs (dict): Dictionary where keys are method names and values are corrected DataFrames.
            sample_index (int or None): If None, plots all samples; otherwise, plots a single sample.

        Returns:
            None: Displays the plot.
        """
        if original_df is None:
            df = list(corrected_dfs.values())[0]
        else:
            df = original_df

        feature_columns = get_feature_columns(df)
        x_vals = [float(col.replace("nm", "")) for col in feature_columns]

        plt.figure(figsize=(10, 4))

        if sample_index is None:
            if original_df is not None:
                # Plot all original spectra
                for i in original_df.index:
                    plt.plot(
                        x_vals,
                        original_df.loc[i, feature_columns],
                        color="gray",
                        alpha=0.4,
                        label="Original" if i == original_df.index[0] else None,
                    )

            # Plot all corrected spectra for each method
            for method, df_corrected in corrected_dfs.items():
                for i in df_corrected.index:
                    plt.plot(
                        x_vals,
                        df_corrected.loc[i, feature_columns],
                        alpha=0.4,
                        label=f"Corrected ({method})"
                        if i == df_corrected.index[0]
                        else None,
                    )

            if title:
                plt.title(title, fontsize=16)
            else:
                plt.title("Baseline Correction Comparison (All Samples)", fontsize=16)

        else:
            if original_df is not None:
                original_spectrum = original_df.loc[sample_index, feature_columns]
                plt.plot(
                    x_vals,
                    original_spectrum,
                    label="Original",
                    linestyle="dashed",
                    linewidth=2,
                )

            for method, df_corrected in corrected_dfs.items():
                corrected_spectrum = df_corrected.loc[sample_index, feature_columns]
                plt.plot(x_vals, corrected_spectrum, label=f"Corrected ({method})")

            plt.title(f"Baseline Correction Comparison for Sample {sample_index}")

        plt.xlabel("Wavelength (nm)", fontsize=14)
        plt.ylabel("Reflectivity", fontsize=14)
        plt.legend(fontsize=12)
        plt.tight_layout()
        if output_name:
            plt.savefig(
                f"{get_working_directory()}/images/corrected_spectra/{output_name}.png"
            )
        plt.show()

    @staticmethod
    def plot_baselines(
        original_df: pd.DataFrame,
        baselines: dict,
        sample_index=None,
        output_name=None,
        title=None,
    ):
        """
        Plots the original spectrum and the estimated baseline for one or all samples.

        Parameters:
            original_df (pd.DataFrame): The original spectral data.
            baselines (dict): A dictionary of estimated baselines.
            sample_index (int or None): If None, plots all samples; otherwise, plots a single sample.

        Returns:
            None: Displays the plot.
        """
        feature_columns = get_feature_columns(original_df)
        x_vals = [float(col.replace("nm", "")) for col in feature_columns]

        plt.figure(figsize=(10, 4))

        # Store legend entries
        legend_entries = []
        legend_colors = []

        if sample_index is None:
            for i in original_df.index:
                plt.plot(
                    x_vals, original_df.loc[i, feature_columns], color="gray", alpha=0.4
                )
            legend_entries.append("Original Spectra")
            legend_colors.append("gray")
        else:
            original_spectrum = original_df.loc[sample_index, feature_columns]
            plt.plot(
                x_vals,
                original_spectrum,
                label="Original Spectrum",
                linestyle="dashed",
                linewidth=2,
                alpha=0.7,
            )
            legend_entries.append("Original Spectrum")
            legend_colors.append("black")

        baseline_labels = set()
        for method, baseline_df in baselines.items():
            if sample_index is None:
                for i in baseline_df.index:
                    plt.plot(
                        x_vals,
                        baseline_df.loc[i, feature_columns],
                        color="blue",
                        linestyle="dashed",
                        alpha=0.2,
                    )
                baseline_labels.add(f"Baseline ({method})")
            else:
                plt.plot(
                    x_vals,
                    baseline_df.loc[sample_index, feature_columns],
                    label=f"Baseline ({method})",
                    linewidth=2,
                    linestyle="dashed",
                    alpha=0.5,
                    color="blue",
                )
                baseline_labels.add(f"Baseline ({method})")

        # Create legend with unique labels and blue color for baselines
        legend_handles = [
            plt.Line2D([0], [0], color="gray", linewidth=2, label="Original Spectra")
        ]
        for label in baseline_labels:
            legend_handles.append(
                plt.Line2D(
                    [0], [0], color="blue", linewidth=2, linestyle="dashed", label=label
                )
            )

        plt.xlabel("Wavelength (nm)", fontsize=14)
        plt.ylabel("Reflectivity", fontsize=14)
        plt.legend(handles=legend_handles, fontsize=12)
        if title:
            plt.title(title, fontsize=16)
        else:
            plt.title(
                "Baselines Visualization"
                + (
                    f" for Sample {sample_index}"
                    if sample_index is not None
                    else " (All Samples)"
                ),
                fontsize=16,
            )
        plt.tight_layout()
        if output_name:
            plt.savefig(f"{get_working_directory()}/images/baseline/{output_name}.png")
        plt.show()

    @staticmethod
    def read_precorrected_reflectance(
        parameters: dict,
    ) -> tuple[pd.DataFrame, pd.DataFrame]:
        """
        Reads the pre-corrected reflectance data from CSV files.

        Parameters:
            parameters (dict): Dictionary containing parameters for reading the data.

        Returns:
            tuple: Tuple containing training and testing DataFrames.
        """
        lam = parameters["lam"]
        p = parameters["p"]
        n_iter = parameters["niter"]

        working_directory_path = get_working_directory()
        output_folder = (
            f"{working_directory_path}/data/{parameters['als_precalculated_folder']}"
        )

        training_filename = f"training_corrected_lam{lam}_p{p}_n{n_iter}.csv"
        testing_filename = f"testing_corrected_lam{lam}_p{p}_n{n_iter}.csv"

        training_output_path = os.path.join(output_folder, training_filename)
        testing_output_path = os.path.join(output_folder, testing_filename)

        training_df = pd.read_csv(training_output_path)
        testing_df = pd.read_csv(testing_output_path)

        training_features = training_df[get_feature_columns(training_df)]
        testing_features = testing_df[get_feature_columns(testing_df)]

        return training_features, testing_features
