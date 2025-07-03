import logging
import itertools
import os
from matplotlib import pyplot as plt
import time
import numpy as np
import pandas as pd
from scipy.signal import find_peaks
from scipy.integrate import simps

from sklearn.cross_decomposition import PLSRegression
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from spectroscopy.src.common.constants import ELEMENT_COLUMNS
from spectroscopy.src.common.utility_functions import get_feature_columns
from spectroscopy import (
    LeafSampleReader,
    DataCleaner,
    BaselineCorrector,
    TargetScaler,
    PLSEstimator,
    PeakFeatureExtractor,
)
from spectroscopy.src.common.utility_functions import (
    get_working_directory,
    get_feature_columns,
    load_config,
    train_test_split,
)

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # --------------------------- Read Data --------------------------- #

    # working_directory_path = get_working_directory()
    # leaf_samples_folder_path = f"{working_directory_path}/data/leaf_samples"
    #
    # leaf_sample_reader = LeafSampleReader(leaf_samples_folder_path)
    #
    # # dried_settings: leaf_states = ['dried'], seasons =  [1,2,3,4]
    # # fresh_settings: leaf_states = ['fresh'], seasons = [1,2,3]
    # leaf_df = leaf_sample_reader.read_selected_csvs(leaf_states=["dried"], seasons=[1,2,3,4])
    # leaf_df = DataCleaner.enforce_data_types(leaf_df)
    #
    # # ------------------------- Data Cleaning ------------------------- #
    #
    # training_df, testing_df = train_test_split_by_season(leaf_df, test_season=1)
    #
    # training_df = DataCleaner.drop_null_data(training_df, row_threshold=0.5, target_col_threshold=0.5,
    #                                          feature_col_threshold=0.5)
    # training_df = DataCleaner.impute_data(training_df, target_method="knn", feature_method="neighbour_avg")
    # training_df = DataCleaner.remove_outliers(training_df)
    #
    # testing_df = DataCleaner.drop_null_data(testing_df, row_threshold=0.5, target_col_threshold=0.5,
    #                                         feature_col_threshold=0.5)
    # testing_df = DataCleaner.impute_data(testing_df, target_method=None, feature_method="neighbour_avg")
    # testing_df = testing_df[training_df.columns] # in case columns were filtered out of the training df
    #
    # # invert reflectivity data for ALS as it is sensitive finding peaks but we suspect that for reflectivity
    # # key points will be drops
    # training_df[get_feature_columns(training_df)] = 1 - training_df[get_feature_columns(training_df)]
    # testing_df[get_feature_columns(testing_df)] = 1 - testing_df[get_feature_columns(testing_df)]
    #
    # print(training_df)
    # print(testing_df)
    #
    # # ------------------------ Target Scaling ------------------------- #
    #
    # # scaling_methods can be applied per column, options: standard, log, minmax, quantile, copula, rank_minmax
    # target_scaler = TargetScaler(scaling_methods={target: 'quantile' for target in ELEMENT_COLUMNS})
    # target_scaler.fit(training_df)
    #
    # training_df = target_scaler.transform(training_df)
    # testing_df = target_scaler.transform(testing_df)
    #
    # print(training_df)
    # print(testing_df)

    # # ---------------------- Define Paths ---------------------- #

    config = load_config("config_example.yml")

    working_directory_path = get_working_directory()
    output_folder = (
        f"{working_directory_path}/data/stratified_als_corrected_samples_fresh"
    )

    leaf_samples_folder_path = (
        f"{working_directory_path}\\{config.get('leaf_samples_folder_path', None)}"
    )
    leaf_sample_reader = LeafSampleReader(leaf_samples_folder_path)

    leaf_df = leaf_sample_reader.read_selected_csvs(
        leaf_states=["fresh"], seasons=[1, 2, 3]
    )
    leaf_df = DataCleaner.enforce_data_types(leaf_df)
    leaf_df = DataCleaner.drop_null_data(
        leaf_df, **config.get("DataCleaner", None).get("drop_null_data", None)
    )

    # ------------------------- Data Cleaning ------------------------- #

    training_df, testing_df = train_test_split(leaf_df, method="stratified")

    train_clean_config = config.get("DataCleaner", None).get("train", None)
    test_clean_config = config.get("DataCleaner", None).get("test", None)

    if train_clean_config.get("enforce_data_types", None) is True:
        training_df = DataCleaner.enforce_data_types(training_df)

    if test_clean_config.get("enforce_data_types", None) is True:
        testing_df = DataCleaner.enforce_data_types(testing_df)

    training_df = DataCleaner.drop_null_data(
        training_df, **train_clean_config.get("drop_null_data", None)
    )

    training_df = DataCleaner.impute_data(
        training_df, **train_clean_config.get("impute_data", None)
    )

    training_df, _ = DataCleaner.remove_outliers(
        training_df, **train_clean_config.get("remove_outliers", None)
    )

    testing_df = DataCleaner.drop_null_data(
        testing_df, **test_clean_config.get("drop_null_data", None)
    )

    testing_df = DataCleaner.impute_data(
        testing_df, **test_clean_config.get("impute_data", None)
    )

    common_columns = training_df.columns.intersection(testing_df.columns)
    training_df = training_df[common_columns]
    testing_df = testing_df[common_columns]

    # convert non-linear reflectance to linear absorbance data
    feature_columns = get_feature_columns(training_df)
    training_df[feature_columns] = np.log(1 / training_df[feature_columns])
    testing_df[feature_columns] = np.log(1 / testing_df[feature_columns])

    lambda_values = [10**3, 10**4, 10**5, 10**6, 10**7]
    # lambda_values = [10 ** 6, 10 ** 7]
    p_values = [0.1, 0.01, 0.001]
    n_iter_values = [5]
    #
    # # ---------------------- Loop Over ALS Parameter Combinations ---------------------- #
    for lam, p, n_iter in itertools.product(lambda_values, p_values, n_iter_values):
        start_time = time.time()
        logging.info(
            f"Applying ALS baseline correction with λ={lam}, p={p}, n_iter={n_iter}"
        )

        # Apply baseline correction
        training_corrected_df, _ = BaselineCorrector.correct_dataframe(
            training_df, method="als", mean_baseline=False, lam=lam, p=p, niter=n_iter
        )
        testing_corrected_df, _ = BaselineCorrector.correct_dataframe(
            testing_df, method="als", mean_baseline=False, lam=lam, p=p, niter=n_iter
        )

        # Generate unique filenames based on parameters
        training_filename = f"training_corrected_lam{lam}_p{p}_n{n_iter}.csv"
        testing_filename = f"testing_corrected_lam{lam}_p{p}_n{n_iter}.csv"

        training_output_path = os.path.join(output_folder, training_filename)
        testing_output_path = os.path.join(output_folder, testing_filename)

        training_corrected_df.to_csv(training_output_path, index=False)
        testing_corrected_df.to_csv(testing_output_path, index=False)

        logging.info(f"Saved corrected training data: {training_output_path}")
        logging.info(f"Saved corrected testing data: {testing_output_path}")

        end_time = time.time()
        logging.info(
            f"time to complete λ={lam}, p={p}, n_iter={n_iter} was {(end_time - start_time):.2f}"
        )
    #
    logging.info("All ALS baseline corrections completed successfully!")

    # Define ALS parameters to search
    # lambda_values = [10 ** 3, 10 ** 4, 10 ** 5, 10 ** 6, 10 ** 7]
    # lambda_values = [10 ** 5]
    # p_values = [0.1]
    # n_iter_values = [5]
    #
    # results = []
    #
    # # Iterate over parameter combinations
    # for lam, p, n_iter in itertools.product(lambda_values, p_values, n_iter_values):
    #     logging.info(f"Testing ALS with λ={lam}, p={p}, n_iter={n_iter}")
    #
    #     training_filename = f"training_corrected_lam{lam}_p{p}_n{n_iter}.csv"
    #     testing_filename = f"testing_corrected_lam{lam}_p{p}_n{n_iter}.csv"
    #
    #     training_output_path = os.path.join(output_folder, training_filename)
    #     testing_output_path = os.path.join(output_folder, testing_filename)
    #
    #     training_corrected_df = pd.read_csv(training_output_path)
    #     testing_corrected_df = pd.read_csv(testing_output_path)
    #
    #     training_targets = LeafSampleReader.extract_targets(training_corrected_df)
    #     training_features = LeafSampleReader.extract_features(training_corrected_df)
    #
    #     testing_targets = LeafSampleReader.extract_targets(testing_corrected_df)
    #     testing_features = LeafSampleReader.extract_features(testing_corrected_df)
    #
    #     # extract summarised peak features
    #     peak_feature_extractor = PeakFeatureExtractor(min_prominence=0.01)
    #     training_features = peak_feature_extractor.fit_transform(training_features, show_plot=True)
    #     testing_features = peak_feature_extractor.transform(testing_features)
    #
    #     pls = PLSEstimator(training_features, training_targets, testing_features, testing_targets)
    #     pls.find_component(components=range(1,40), folds=10)
    #     pls.fit_predict()
    #
    #     testing_predictions = pls.y_pred
    #     validation_rmse = pls.validation_rmse
    #     num_components = pls.n_components
    #
    #     # Store results
    #     results.append({
    #         "lambda": lam,
    #         "p": p,
    #         "n_iter": n_iter,
    #         "num_components": num_components,
    #         "validation_rmse": validation_rmse
    #     })
    #
    # # Convert results to DataFrame
    # df_results = pd.DataFrame(results)
    # df_results.to_csv(f"{get_working_directory()}/data/results/als_baseline_correction_tuning_1.csv", header=False)
