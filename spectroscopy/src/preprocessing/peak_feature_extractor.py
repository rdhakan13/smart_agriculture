import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from scipy.integrate import simps
from spectroscopy.src.common.utility_functions import get_feature_columns


class PeakFeatureExtractor:
    """
    Class for detecting and applying peak-based feature extraction on spectral data.
    This class identifies peaks in the mean reflectivity curve of training data
    and extracts features based on these peaks.
    """

    def __init__(self, min_prominence=0.01):
        """
        Class for detecting and applying peak-based feature extraction.

        Parameters:
            min_prominence: Minimum prominence to detect peaks in training data.
        """
        self.min_prominence = min_prominence
        self.peak_regions = (
            None  # Stores peak wavelength regions found in training data
        )
        self.wavelengths = None  # Stores wavelength order for consistency

    def _extract_numeric_wavelengths(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, np.ndarray]:
        """
        Extracts numeric wavelengths from column names (e.g., '400nm' -> 400.0).

        Parameters:
            df: DataFrame where each row is a sample, columns are reflectivity values across wavelengths.

        Returns:
            feature_columns: List of feature column names (wavelengths).
            numeric_wavelengths: Numpy array of numeric wavelengths.
        """
        feature_columns = get_feature_columns(df)
        numeric_wavelengths = np.array(
            [float(re.sub(r"[^0-9.]", "", col)) for col in feature_columns]
        )
        return feature_columns, numeric_wavelengths

    def fit(self, df: pd.DataFrame, show_plot: bool = True):
        """
        Fit the peak detection model on training data and optionally plot detected peaks.

        Parameters:
            df: DataFrame where each row is a sample, columns are reflectivity values across wavelengths.
            show_plot: If True, plots detected peaks for verification.

        Returns:
            None
        """
        # Step 1: Extract numeric wavelengths and sort columns
        feature_columns, numeric_wavelengths = self._extract_numeric_wavelengths(df)
        sorted_indices = np.argsort(numeric_wavelengths)
        self.wavelengths = numeric_wavelengths[sorted_indices]
        df_sorted = df[feature_columns].iloc[
            :, sorted_indices
        ]  # Sort columns by wavelength

        # Step 2: Compute mean reflectivity curve across all training samples
        mean_reflectivity = df_sorted.mean(axis=0)

        # Step 3: Identify peaks on the mean reflectivity curve
        peaks, properties = find_peaks(
            mean_reflectivity, prominence=self.min_prominence
        )

        # Step 4: Store detected peak regions
        self.peak_regions = []
        for peak in peaks:
            peak_wavelength = self.wavelengths[peak]
            peak_width = (
                properties["widths"][np.where(peaks == peak)][0]
                if "widths" in properties
                else 20
            )  # Default width

            # Define left and right boundaries of peak region
            left_idx = max(0, peak - int(peak_width))  # Left boundary
            right_idx = min(
                len(self.wavelengths) - 1, peak + int(peak_width)
            )  # Right boundary

            self.peak_regions.append(
                (
                    peak_wavelength,
                    self.wavelengths[left_idx],
                    self.wavelengths[right_idx],
                )
            )

        print(f"✅ Found {len(self.peak_regions)} peak regions in training data.")

        # Step 5: Plot detected peaks for verification
        if show_plot:
            self.plot_peaks(mean_reflectivity, peaks)

    def get_num_peak_regions(self) -> int:
        """
        Returns the number of detected peak regions.

        Returns:
            int: Number of detected peak regions.
        """
        return len(self.peak_regions)

    def plot_peaks(self, mean_reflectivity, peaks):
        """
        Plots the mean reflectivity curve with detected peaks.

        Parameters:
            mean_reflectivity: The mean reflectivity values (numpy array).
            peaks: Indices of detected peaks (from find_peaks).

        Returns:
            None
        """
        plt.figure(figsize=(10, 5))
        plt.plot(
            self.wavelengths, mean_reflectivity, label="Mean Reflectivity", color="blue"
        )
        plt.scatter(
            self.wavelengths[peaks],
            mean_reflectivity[peaks],
            color="red",
            label="Detected Peaks",
            zorder=3,
        )

        # Highlight peak regions
        for peak_wavelength, left_wavelength, right_wavelength in self.peak_regions:
            plt.axvspan(
                left_wavelength, right_wavelength, color="orange", alpha=0.3, label=None
            )

        plt.xlabel("Wavelength (nm)")
        plt.ylabel("Reflectivity")
        plt.title("Detected Peaks in Mean Reflectivity")
        plt.legend()
        plt.grid(True)
        plt.show()

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply the stored peak regions to extract features for new data.

        Parameters:
            df: DataFrame with reflectivity values (same structure as training data).

        Returns:
            DataFrame with extracted peak features.
        """
        if self.peak_regions is None:
            raise ValueError(
                "Model has not been fitted yet. Call `fit()` on training data first."
            )

        # Step 1: Extract numeric wavelengths and sort columns
        feature_columns, numeric_wavelengths = self._extract_numeric_wavelengths(df)
        sorted_indices = np.argsort(numeric_wavelengths)
        df_sorted = df[feature_columns].iloc[
            :, sorted_indices
        ]  # Sort columns by wavelength

        # Step 2: Extract features based on stored peak regions
        extracted_features = []
        for index, row in df_sorted.iterrows():
            sample_features = {}

            for i, (peak_wavelength, left_wavelength, right_wavelength) in enumerate(
                self.peak_regions
            ):
                # Find indices of the peak region
                left_idx = np.argmin(np.abs(self.wavelengths - left_wavelength))
                right_idx = np.argmin(np.abs(self.wavelengths - right_wavelength))

                # Extract reflectivity values in the peak region
                region_reflectivity = row.iloc[left_idx : right_idx + 1].values

                # Compute peak-based features
                sample_features[f"peak_{i + 1}_max"] = np.max(
                    region_reflectivity
                )  # Peak height
                sample_features[f"peak_{i + 1}_min"] = np.min(
                    region_reflectivity
                )  # Min reflectivity
                sample_features[f"peak_{i + 1}_auc"] = simps(
                    region_reflectivity, self.wavelengths[left_idx : right_idx + 1]
                )  # Area under curve

            extracted_features.append(sample_features)

        # Convert extracted features into a DataFrame
        feature_df = pd.DataFrame(extracted_features, index=df.index)

        return feature_df

    def fit_transform(self, df, show_plot=True):
        """
        Fit the model on the dataset and return transformed features.

        Parameters:
        - df: DataFrame with reflectivity values.
        - show_plot: If True, plots detected peaks.

        Returns:
        - DataFrame with extracted peak features.
        """
        self.fit(df, show_plot=show_plot)
        return self.transform(df)
