import logging

import numpy as np
from spectroscopy import LeafSampleReader, DataCleaner, BaselineCorrector
from spectroscopy.src.common.utility_functions import (
    get_working_directory,
    get_feature_columns,
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    leaf_samples_folder_path = f"{get_working_directory()}/data/leaf_samples"

    leaf_sample_reader = LeafSampleReader(leaf_samples_folder_path)
    leaf_df = leaf_sample_reader.read_all_csvs(leaf_state="dried", season=1)

    leaf_df = DataCleaner.enforce_data_types(leaf_df)
    leaf_df = DataCleaner.drop_null_data(
        leaf_df, row_threshold=0.5, target_col_threshold=0.5, feature_col_threshold=0.5
    )
    leaf_df = DataCleaner.impute_data(
        leaf_df, target_method=None, feature_method="neighbour_avg"
    )

    # uncomment to try different baseline generating techniques to plot against each other
    method_mapping = [
        # ("polynomial", {"poly_order": 2}),
        ("polynomial", {"poly_order": 3}),
        # ("polynomial", {"poly_order": 4}),
        # ("polynomial", {"poly_order": 5}),
        # ("wavelet", {"wavelet": "db4", "level": 1}),
        # ("wavelet", {"wavelet": "db4", "level": 2}),
        # ("wavelet", {"wavelet": "haar", "level": 1}),
        # ("wavelet", {"wavelet": "haar", "level": 2}),
        # ("iterative_mean", {"iterations": 1}),
        # ("iterative_mean", {"iterations": 2}),
        # ("iterative_mean", {"iterations": 3}),
        # ("iterative_mean", {"iterations": 4}),
        # ("iterative_mean", {"iterations": 5}),
        # ("ica", {"n_components": 5}),
        # ("ica", {"n_components": 5}),
        # ("ica", {"n_components": 5}),
        # ("ica", {"n_components": 5}),
        # ("als", {"lam": 1e4, "p": 0.01, "niter": 5}),
        # ("als", {"lam": 1e3, "p": 0.01, "niter": 5}),
        # ("als", {"lam": 25e4, "p": 0.01, "niter": 10}),
        # ("als", {"lam": 1e5, "p": 0.01, "niter": 10}),
        # ("als", {"lam": 25e5, "p": 0.01, "niter": 10}),
        # ("als", {"lam": 100e5, "p": 0.01, "niter": 10}),
    ]

    corrected_dfs = {}
    baselines = {}

    feature_columns = get_feature_columns(leaf_df)
    leaf_df[feature_columns] = np.log(1 / leaf_df[feature_columns])

    for method, param_dict in method_mapping:
        logging.info(f"Fitting {method} with params: {param_dict}")
        corrected_df, baseline = BaselineCorrector.correct_dataframe(
            leaf_df, method=method, mean_baseline=False, **param_dict
        )

        param_name = (
            "_".join(f"{k}-{v}" for k, v in param_dict.items())
            if param_dict
            else "default"
        )
        key_name = f"{method}_{param_name}"

        corrected_dfs[key_name] = corrected_df
        baselines[key_name] = baseline

    BaselineCorrector.plot_baselines(
        leaf_df,
        baselines,
        sample_index=None,
        output_name="Samples and Fitted Baselines using 3-degree Polynomial",
        title="Samples and Fitted Baselines using 3-degree Polynomial",
    )
    BaselineCorrector.compare_correction(
        original_df=None,
        corrected_dfs=corrected_dfs,
        sample_index=None,
        output_name="Baseline Corrected Samples using 3-degree Polynomial",
        title="Baseline Corrected Samples using 3-degree Polynomial",
    )

    # output cleaned data samples
    # output_path = f"{get_working_directory()}/data/cleaned_samples/Dried_season01.csv"
    # corrected_df.to_csv(output_path)
