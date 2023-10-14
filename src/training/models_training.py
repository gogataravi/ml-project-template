import pickle
from typing import Any, List, Literal, Optional, Tuple, Union

import numpy as np
from sklearn import decomposition, metrics
from sklearn.base import ClassifierMixin
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier, StackingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline

from utils.ml_logging import get_logger

# Set up logging
logger = get_logger()


class ModelTrainer:
    """
    Class for training ensemble classifiers using randomized search.

    Parameters:
        X_train (array-like): Training data.
        y_train (array-like): Target labels for training data.
        random_state (int): Random seed for reproducibility (default=1).

    Methods:
        run_training: Train a specified Classifier.
        _make_scoring: Creates a scorer based on the provided type.
        _perform_randomized_search: Performs randomized search for hyperparameter tuning.
    """

    def __init__(self, X_train: np.array, y_train: np.array, random_state: int = 1):
        self.X_train = X_train
        self.y_train = y_train
        self.random_state = random_state

    def run_hyperparameter_opt(
        self,
        estimator: Union[AdaBoostClassifier, BaggingClassifier],
        scorer: Literal["Recall"] = "Recall",
        parameters: dict = {},
        n_jobs: int = -1,
        n_iter_search: int = 500,
        apply_pca: Optional[bool] = None,
    ) -> Any:
        """
        Train a specified classifier with or without PCA.

        Parameters:
            estimator: AdaBoostClassifier or BaggingClassifier.
            scorer: The scorer type. Currently only supports 'Recall'.
            parameters: Dictionary of parameters to be used for RandomizedSearchCV.
            n_jobs: Number of jobs to run in parallel.
            n_iter_search: Number of parameter settings that are sampled.
            apply_pca: Whether to apply PCA before training the classifier.

        Returns:
            tuned_estimator: The estimator trained with the best found parameters.
        """
        try:
            logger.info(
                f"Starting training for {type(estimator).__name__} with scorer {scorer}."
            )

            if apply_pca:
                pca = decomposition.PCA()
                pipe = Pipeline(
                    steps=[("pca", pca), (str(type(estimator).__name__), estimator)]
                )
                logger.info("PCA is applied before training.")
            else:
                pipe = Pipeline(steps=[(str(type(estimator).__name__), estimator)])

            grid_obj = self._perform_randomized_search(
                pipe, parameters, scorer, n_jobs, n_iter_search
            )
            tuned_estimator = grid_obj.best_estimator_
            tuned_estimator.fit(self.X_train, self.y_train)

            logger.info(f"Training successful for {type(estimator).__name__}.")
            return tuned_estimator

        except Exception as e:
            logger.error(
                f"Error occurred during training for {type(estimator).__name__}: {e}"
            )
            raise e

    def _make_scoring(self, scorer_type: Literal["Recall"]) -> Any:
        """
        Create a scorer based on the provided type.

        Parameters:
            scorer_type: Type of scorer to be created.

        Returns:
            acc_scorer: Scorer function.
        """
        try:
            logger.info(f"Creating scorer of type {scorer_type}.")

            if scorer_type == "Recall":
                acc_scorer = metrics.make_scorer(metrics.recall_score)
                logger.info(f"Scorer of type {scorer_type} created successfully.")
                return acc_scorer

            else:
                logger.warning(
                    f"Scorer type {scorer_type} not recognized. Returning None."
                )
                return None

        except Exception as e:
            logger.error(
                f"Error occurred while creating scorer of type {scorer_type}: {e}"
            )
            raise e

    def _perform_randomized_search(
        self,
        estimator: Any,
        parameters: dict,
        scorer: Literal["Recall"],
        n_jobs: int = -1,
        n_iter_search: int = 500,
        cv: int = 5,
    ) -> Any:
        """
        Perform randomized search for hyperparameter tuning.

        Parameters:
            estimator: Classifier to tune.
            parameters: Hyperparameters and their possible values.
            scorer: Scorer type to be used.
            n_jobs: Number of jobs to run in parallel.
            n_iter_search: Number of parameter settings that are sampled.
            cv: Number of folds in cross-validation.

        Returns:
            grid_obj: Trained RandomizedSearchCV object.
        """
        try:
            logger.info(f"Initiating Randomized Search for {estimator}.")
            acc_scorer = self._make_scoring(scorer)
            grid_obj = RandomizedSearchCV(
                estimator,
                n_iter=n_iter_search,
                param_distributions=parameters,
                scoring=acc_scorer,
                cv=cv,
                n_jobs=n_jobs,
            )
            grid_obj.fit(self.X_train, self.y_train)
            logger.info(
                f"Randomized Search successful for {estimator}. Best parameters: {grid_obj.best_params_}"
            )
            return grid_obj
        except Exception as e:
            logger.error(
                f"Error occurred during Randomized Search for {estimator}: {e}"
            )
            raise e

    def stack_and_fit_models(
        self,
        base_models: List[Tuple[str, ClassifierMixin]],
        meta_model: ClassifierMixin,
        X_train: np.array,
        y_train: np.array,
        refit_base_models: bool = False,
    ) -> StackingClassifier:
        """
        Stack the given base models, fit the StackingClassifier on the training data, and return the fitted classifier.

        Parameters:
        base_models (List[Tuple[str, ClassifierMixin]]): List of (name, model) tuples.
        meta_model (ClassifierMixin): Meta-model for stacking.
        X_train (np.array): Training data.
        y_train (np.array): Training labels.
        refit_base_models (bool): Flag indicating whether to refit base models.

        Returns:
        StackingClassifier: Fitted StackingClassifier.
        """
        try:
            if refit_base_models:
                logger.info("Refitting base models.")
                # Extract parameters from base models, instantiate new copies
                new_base_models = [
                    (name, model.__class__(**model.get_params()))
                    for name, model in base_models
                ]
                base_models = new_base_models

            # Instantiate and fit StackingClassifier
            logger.info("Instantiating and fitting StackingClassifier.")
            stacking_model = StackingClassifier(
                estimators=base_models,
                final_estimator=meta_model,
                passthrough=True,
                cv=5,
                verbose=2,
            )
            stacking_model.fit(X_train, y_train)
            logger.info("StackingClassifier fitted successfully.")
            return stacking_model

        except Exception as e:
            logger.error(f"Error in stack_and_fit_models: {e}")
            raise e

    def save_model_to_pickle(estimator: Any, file_path: str) -> None:
        """
        Save a trained model to a specified file using Pickle.

        Parameters:
            estimator (Any): The trained model to save.
            file_path (str): The path to the file where the model will be saved.
        """
        try:
            logger.info(f"Saving model to {file_path}.")

            with open(file_path, "wb") as file:
                pickle.dump(estimator, file)

            logger.info(f"Model saved successfully to {file_path}.")

        except Exception as e:
            logger.error(f"Error occurred while saving model to {file_path}: {e}")
            raise e
