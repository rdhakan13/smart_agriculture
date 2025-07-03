import logging
import GPy
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime as dt
from sklearn.svm import SVR
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
from sklearn.model_selection import KFold, GridSearchCV
from sklearn.linear_model import LinearRegression, MultiTaskElasticNet, MultiTaskLasso
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import cross_val_score
from sklearn.multioutput import MultiOutputRegressor
from catboost import CatBoostRegressor


class RegressionModels:
    """
    A class for regression models with hyperparameter tuning and cross-validation.
    """

    def __init__(self, model_type="GPR", k_folds=5, random_state=42):
        """
        Initialize the regression model with specified type and parameters.

        Parameters:
            model_type: Type of regression model to use. Options include:
                - "LR": Linear Regression
                - "MLR": Multi-Output Linear Regression
                - "PLS": Partial Least Squares Regression
                - "GPR": Gaussian Process Regression
                - "GPCO": Gaussian Process Co-Regression
                - "SVR": Support Vector Regression
                - "forest": Random Forest Regressor
                - "catboost": CatBoost Regressor
                - "MTElasticNet": Multi-Task Elastic Net
                - "MTLasso": Multi-Task Lasso
                - "GradientBoost": Gradient Boosting Regressor
                - None: No model (for custom use cases)
            k_folds: Number of folds for cross-validation.
            random_state: Random state for reproducibility.
        """
        self.model_type = model_type
        self.k_folds = k_folds
        self.random_state = random_state
        self.model = None
        self.grid_search = None
        self.kf = KFold(n_splits=k_folds, shuffle=True, random_state=random_state)
        self.avg_validation_score = None

    def fit(self, X, y, custom_param_grid: dict = None):
        """
        Fit the regression model to the training data.

        Parameters:
            X: Feature matrix (input data).
            y: Target vector (output data).
            custom_param_grid: Custom parameter grid for hyperparameter tuning.

        Returns:
            None
        """
        model = None
        param_grid = None
        if self.model_type == "LR":
            param_grid = {}
            model = LinearRegression()
        elif self.model_type == "MLR":
            param_grid = {}
            model = MultiOutputRegressor(LinearRegression())
        elif self.model_type == "PLS":
            param_grid = {"n_components": [2, 5, 10]}
            model = PLSRegression()
        elif self.model_type == "GPR":
            param_grid = {
                "kernel": [C(1.0) * RBF(l) for l in [0.1, 1.0, 10.0]],
                "alpha": [1e-10, 1e-5, 1e-2],
            }
            model = GaussianProcessRegressor(
                normalize_y=True, random_state=self.random_state
            )
        elif self.model_type == "GPCO":
            kernel = GPy.kern.RBF(input_dim=X.shape[1])
            self.model = GPy.models.GPCoregionalizedRegression(X, y, kernel)
            self.model.optimize()
            return
        elif self.model_type == "SVR":
            param_grid = {
                "estimator__C": [0.1, 1, 10],
                "estimator__epsilon": [0.01, 0.1, 1],
                "estimator__kernel": [
                    "rbf",
                    "linear",
                ],  # Optional: include if you want to tune kernels
            }
            model = MultiOutputRegressor(SVR())
        elif self.model_type == "forest":
            param_grid = {
                "estimator__n_estimators": [100, 200],
                "estimator__max_depth": [None, 10, 20],
                "estimator__min_samples_split": [2, 5],
                "estimator__min_samples_leaf": [1, 2],
            }
            model = MultiOutputRegressor(RandomForestRegressor())
        elif self.model_type == "catboost":
            param_grid = {
                "depth": [4, 6, 8],  # Controls tree depth (complexity)
                "learning_rate": [0.01, 0.05, 0.1],  # Step size
                "iterations": [100, 300, 500],  # Number of trees
                "l2_leaf_reg": [1, 3, 5],  # L2 regularization on leaf weights
                "bagging_temperature": [0, 1, 5],  # Controls randomness in bagging
                "random_strength": [1, 5, 10],  # Randomness in feature splits
            }
            model = CatBoostRegressor(
                loss_function="MultiRMSE", iterations=500, depth=6
            )
        elif self.model_type == "MTElasticNet":
            param_grid = {"alpha": [0.1, 1, 10], "l1_ratio": [0.2, 0.5, 0.8]}
            model = MultiTaskElasticNet()
        elif self.model_type == "MTLasso":
            param_grid = {"alpha": [0.1, 1, 10]}
            model = MultiTaskLasso()
        elif self.model_type == "GradientBoost":
            param_grid = {
                "n_estimators": [50, 100, 200],
                "learning_rate": [0.01, 0.1, 0.2],
            }
            model = GradientBoostingRegressor()
        elif self.model_type is None:
            pass
        else:
            raise ValueError("Invalid model_type. Choose a valid model")

        # Update param_grid if custom_param_grid is provided
        if custom_param_grid and model not in ["LR", "MLR"]:
            param_grid.update(custom_param_grid)

        if (
            param_grid and self.model_type != "MLR"
        ):  # No grid search for simple linear regression
            start = dt.now()
            self.grid_search = GridSearchCV(
                model,
                param_grid,
                cv=self.kf,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
            )
            self.grid_search.fit(X, y)
            logging.info(
                f"GridSearchCV completed in {(dt.now() - start).total_seconds()} seconds"
            )
            self.model = self.grid_search.best_estimator_
            logging.info(f"Best params: {self.grid_search.best_params_}")
            self.avg_validation_score = -self.grid_search.best_score_
        else:
            self.model = model
            self.model.fit(X, y)
            cv_scores = cross_val_score(
                self.model,
                X,
                y,
                cv=self.kf,
                scoring="neg_root_mean_squared_error",
                n_jobs=-1,
            )
            self.avg_validation_score = -np.mean(cv_scores)

    def predict(self, X) -> tuple:
        """
        Predict using the fitted model.

        Parameters:
            X: Feature matrix (input data).

        Returns:
            y_pred: Predicted values.
            y_std: Standard deviation of predictions (if applicable).
        """
        if self.model_type in [
            "MLR",
            "PLS",
            "GPR",
            "SVR",
            "MTElasticNet",
            "MTLasso",
            "GradientBoost",
            "forest",
            "catboost",
        ]:
            y_pred = self.model.predict(X)
            y_std = None
        elif self.model_type == "GPCO":
            y_pred, y_std = self.model.predict(X)
        elif self.model_type is None:
            y_pred = None
            y_std = None
        else:
            raise ValueError("Invalid model_type. Choose a valid model")
        return y_pred, y_std

    def get_grid_search_results(self):
        """
        Get the results of the grid search.

        Parameters:
            None

        Returns:
            cv_results: Cross-validation results from the grid search.
        """
        if self.grid_search:
            return self.grid_search.cv_results_
        else:
            raise ValueError(
                "No grid search performed. Fit the model with parameter tuning first."
            )

    def plot_grid_search_results(self, save_path: str = None):
        """
        Plot the results of the grid search.

        Parameters:
            save_path: Path to save the plot. If None, display the plot.

        Returns:
            None
        """
        if self.grid_search:
            results = self.grid_search.cv_results_
            param_names = [str(param) for param in results["params"]]
            neg_rmse_scores = results["mean_test_score"]

            plt.figure(figsize=(8, 5))
            plt.plot(
                param_names,
                neg_rmse_scores,
                marker="o",
                linestyle="dashed",
                label="Negative RMSE",
            )
            plt.xlabel("Parameter Setting")
            plt.ylabel("Negative Root Mean Squared Error")
            plt.xticks(rotation=45, ha="right")
            plt.title("Grid Search Performance")
            plt.legend()
            plt.tight_layout()
            if save_path:
                plt.savefig(save_path)
            else:
                plt.show()
        else:
            logging.info(
                "No grid search performed. Fit the model with parameter tuning first."
            )

    def get_best_params(self):
        """
        Get the best parameters from the grid search.

        Parameters:
            None

        Returns:
            best_params: Best parameters from the grid search.
        """
        if self.grid_search:
            return self.grid_search.best_params_
        else:
            logging.info(
                "No grid search performed. Fit the model with parameter tuning first."
            )

    def get_best_avg_validation_score(self):
        """
        Get the best average validation score from the grid search.

        Parameters:
            None

        Returns:
            avg_validation_score: Best average validation score from the grid search.
        """
        return self.avg_validation_score
