from spectroscopy import (
    LeafSampleReader,
    DataCleaner,
    PeakFeatureExtractor,
    DataReducer,
    TargetScaler,
    BaselineCorrector,
    Metrics,
    PerformancePlotter,
    RegressionModels,
)
from spectroscopy.src.common.utility_functions import (
    get_feature_columns,
    get_working_directory,
    train_test_split,
    load_config,
    create_timestamp_filename,
)
from tabulate import tabulate
import logging
import os
import pandas as pd
import itertools
import time
import numpy as np

if __name__ == "__main__":
    # ----------------------------- Settings ---------------------------- #

    config = load_config("als_config.yml")

    settings = config.get("settings", None)

    logging.basicConfig(level="INFO")

    working_directory_path = get_working_directory()

    leaf_samples_folder_path = f"{working_directory_path}/data/leaf_samples"

    # ---------------------------- Read Data ---------------------------- #

    read_data = config.get("read_data", None)

    leaf_sample_reader = LeafSampleReader(leaf_samples_folder_path)

    leaf_df = leaf_sample_reader.read_selected_csvs(
        **read_data.get("LeafSampleReader", None).get("read_selected_csvs", None)
    )

    leaf_df = DataCleaner.enforce_data_types(leaf_df)

    if read_data.get("DataCleaner", None).get("drop_null_data", None) is not None:
        leaf_df = DataCleaner.drop_null_data(
            leaf_df, **read_data.get("DataCleaner", None).get("drop_null_data", None)
        )

    if read_data.get("DataCleaner", None).get("impute_data", None) is not None:
        leaf_df = DataCleaner.impute_data(
            leaf_df, **read_data.get("DataCleaner", None).get("impute_data", None)
        )

    if read_data.get("DataCleaner", None).get("remove_outliers", None) is not None:
        leaf_df, _ = DataCleaner.remove_outliers(
            leaf_df, **read_data.get("DataCleaner", None).get("remove_outliers", None)
        )

    # ------------------------- Test Train Split ------------------------ #

    test_train_split_config = config.get("TestTrainSplit", None)
    training_df, testing_df = train_test_split(
        leaf_df, test_train_split_config.get("method", "stratified")
    )

    # -------------------------- Data Cleaning -------------------------- #

    train_clean_config = test_train_split_config.get("train", None)
    test_clean_config = test_train_split_config.get("test", None)

    if train_clean_config is not None:
        if (
            train_clean_config.get("DataCleaner").get("drop_null_data", None)
            is not None
        ):
            training_df = DataCleaner.drop_null_data(
                training_df, **train_clean_config.get("drop_null_data", None)
            )
        if train_clean_config.get("DataCleaner").get("impute_data", None) is not None:
            training_df = DataCleaner.impute_data(
                training_df,
                **train_clean_config.get("DataCleaner").get("impute_data", None),
            )
        if (
            train_clean_config.get("DataCleaner").get("remove_outliers", None)
            is not None
        ):
            training_df, _ = DataCleaner.remove_outliers(
                training_df,
                **train_clean_config.get("DataCleaner").get("remove_outliers", None),
            )

    if test_clean_config is not None:
        if test_clean_config.get("DataCleaner").get("drop_null_data", None) is not None:
            testing_df = DataCleaner.drop_null_data(
                testing_df,
                **test_clean_config.get("DataCleaner").get("drop_null_data", None),
            )
        if test_clean_config.get("DataCleaner").get("impute_data", None) is not None:
            testing_df = DataCleaner.impute_data(
                testing_df,
                **test_clean_config.get("DataCleaner").get("impute_data", None),
            )

    common_columns = training_df.columns.intersection(testing_df.columns)

    training_df = training_df[common_columns]
    testing_df = testing_df[common_columns]

    training_df.reset_index(inplace=True)
    testing_df.reset_index(inplace=True)

    # ------------------------ Feature Selection ----------------------- #

    training_targets = LeafSampleReader.extract_targets(training_df)
    training_features = LeafSampleReader.extract_features(training_df)

    testing_targets = LeafSampleReader.extract_targets(testing_df)
    testing_features = LeafSampleReader.extract_features(testing_df)

    # ------------------------- Target Scaling -------------------------- #

    target_scaler_config = config.get("TargetScaler", None)

    if target_scaler_config is not None:
        target_scaler = TargetScaler(**target_scaler_config)
        target_scaler.fit(training_targets)

        training_targets = target_scaler.transform(training_targets)
        testing_targets = target_scaler.transform(testing_targets)

    # ----------------------- Baseline Correction ---------------------- #

    # lambda_values = [10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7]
    # p_values = [0.1, 0.01, 0.001]
    # n_iter_values = [5]
    # min_prominence_values = [0.1, 0.01, 0.001]

    lambda_values = [10**5]
    p_values = [0.1]
    n_iter_values = [5]
    min_prominence_values = [0.01]

    original_training_features = training_features.copy()
    original_testing_features = testing_features.copy()

    for min_prominence, lam, p, n_iter in itertools.product(
        min_prominence_values, lambda_values, p_values, n_iter_values
    ):
        parameters = {
            "lam": lam,
            "p": p,
            "niter": n_iter,
            "als_precalculated_folder": f"stratified_als_corrected_samples_dried",
        }

        training_features = original_training_features.copy()
        testing_features = original_testing_features.copy()

        feature_columns = get_feature_columns(training_df)
        # training_df[feature_columns] = np.log(1 / training_df[feature_columns])
        # testing_df[feature_columns] = np.log(1 / testing_df[feature_columns])

        # # ---------------------- Loop Over ALS Parameter Combinations ---------------------- #

        baseline_correction_config = config.get("BaselineCorrector", {})

        if baseline_correction_config is not None:
            if baseline_correction_config.get("method", None) == "als":
                training_features, testing_features = (
                    BaselineCorrector.read_precorrected_reflectance(parameters)
                )
            else:
                training_features, _ = BaselineCorrector.correct_dataframe(
                    training_features,
                    baseline_correction_config.get("method", None),
                    **baseline_correction_config.get("parameters", {}),
                )
                testing_features, _ = BaselineCorrector.correct_dataframe(
                    testing_features,
                    baseline_correction_config.get("method", None),
                    **baseline_correction_config.get("parameters", {}),
                )

        peak_feature_extractor = PeakFeatureExtractor(min_prominence=min_prominence)
        training_features = peak_feature_extractor.fit_transform(
            training_features, show_plot=True
        )
        testing_features = peak_feature_extractor.transform(testing_features)
        num_peaks = peak_feature_extractor.get_num_peak_regions()

        # ------------------------- Data Reduction ------------------------- #

        data_reducer_config = config.get("DataReducer", None)
        if data_reducer_config is not None:
            data_reducer = DataReducer(
                config.get("DataReducer", None).get("method", None)
            )
            training_features = data_reducer.reduce_data(
                training_features,
                **config.get("DataReducer", None).get("parameters", None),
            )
            testing_features = data_reducer.reduce_data(
                testing_features,
                **config.get("DataReducer", None).get("parameters", None),
            )

        # ---------------------------- Ml Model ---------------------------- #
        # testing_targets = testing_targets[['B','Mg','K']]

        regression_model_config = config.get("RegressionModels", None)
        model_type = regression_model_config.get("model_type", None)

        regression_model = RegressionModels(
            regression_model_config.get("model_type", None),
            regression_model_config.get("k_folds", 5),
        )

        custom_param_grid = config.get("RegressionModels", None).get("param_grid", None)
        # max_components = min(len(training_features.columns),77)
        # custom_param_grid.update({'n_components': range(1, max_components+1)})

        regression_model.fit(
            training_features, training_targets, custom_param_grid=custom_param_grid
        )

        best_params = regression_model.get_best_params()
        # regression_model.plot_grid_search_results(
        #     save_path=f"{results_path + model_type}_grid_search.png") # use argument save_path="" to save output image

        # print(parameters)
        print(
            f"Avg Validation RMSE Score {regression_model.get_best_avg_validation_score():.5f}"
        )
        print(
            f"{lam},{p},{n_iter},{min_prominence},{num_peaks},{best_params},{regression_model.get_best_avg_validation_score():.5f}"
        )

        testing_predictions, _ = regression_model.predict(testing_features)
        # print(regression_model.get_best_params())
        #
        # # --------------------- Performance Extraction --------------------- #
        #
        tbl_args = {"headers": "keys", "tablefmt": "simple", "floatfmt": ".2f"}
        #
        testing_metrics = Metrics(
            testing_predictions, testing_targets, config.get("Metrics", None)
        )
        testing_metrics.include_element_units()
        print(tabulate(testing_metrics.results, **tbl_args))
        # # testing_metrics.results.to_csv(create_timestamp_filename(
        # #     prefix=f"{results_path}TEST_{model_type}"))
        #
        # # ---------------------- Performance Plotting ---------------------- #
        #
        # PerformancePlotter.plot_predictions(
        #     testing_targets, testing_predictions, "Targets v Predictions")
        # PerformancePlotter.plot_residuals(
        #     testing_targets, testing_predictions, title=f"{model_type} Model Residuals")
