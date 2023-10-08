import numpy as np
from scipy.stats import randint as sp_randint
from sklearn import metrics
from sklearn.ensemble import AdaBoostClassifier, BaggingClassifier
from sklearn.model_selection import RandomizedSearchCV
from sklearn.pipeline import Pipeline
from sklearn.tree import DecisionTreeClassifier


class ModelTrainer:
    """
    Class for training ensemble classifiers using randomized search.

    Parameters:
        X_train (array-like): Training data.
        y_train (array-like): Target labels for training data.
        random_state (int): Random seed for reproducibility (default=1).
        n_jobs (int): Number of CPU cores to use for parallel processing (default=-1).
        n_iter_search (int): Number of iterations for randomized search (default=500).

    Attributes:
        acc_scorer (callable): Scorer for accuracy metric.

    Methods:
        train_bagging_classifier: Train a Bagging Classifier.
        train_ada_boost_classifier: Train an AdaBoost Classifier.
    """

    def __init__(self, X_train, y_train, random_state=1, n_jobs=-1, n_iter_search=500):
        self.X_train = X_train
        self.y_train = y_train
        self.random_state = random_state
        self.n_jobs = n_jobs
        self.n_iter_search = n_iter_search
        self.acc_scorer = metrics.make_scorer(metrics.recall_score)

    def train_bagging_classifier(self):
        """
        Train a Bagging Classifier.

        Returns:
            tuned_estimator (BaggingClassifier): Trained Bagging Classifier.
        """
        estimator = BaggingClassifier(
            random_state=self.random_state, n_jobs=self.n_jobs
        )
        parameters = {
            "max_samples": [0.1, 0.6, 0.9, 1],
            "max_features": [0.1, 0.6, 0.8, 0.9, 1],
            "n_estimators": [10, 20, 40, 50, 100],
        }
        grid_obj = self._perform_randomized_search(estimator, parameters)
        tuned_estimator = grid_obj.best_estimator_
        tuned_estimator.fit(self.X_train, self.y_train)
        return tuned_estimator

    def train_ada_boost_classifier(self):
        """
        Train an AdaBoost Classifier.

        Returns:
            tuned_estimator (AdaBoostClassifier): Trained AdaBoost Classifier.
        """
        base_estimator = AdaBoostClassifier(random_state=self.random_state)
        pipe = Pipeline([("estimator", base_estimator)])
        parameters = {
            "estimator__base_estimator": [
                DecisionTreeClassifier(
                    max_depth=depth,
                    max_leaf_nodes=9,
                    min_impurity_decrease=0.001,
                    min_samples_leaf=leafs,
                    random_state=self.random_state,
                )
                for depth in [3, 5, 8]
                for leafs in [8, 10]
            ],
            "estimator__n_estimators": sp_randint(10, 110),
            "estimator__learning_rate": np.arange(0.01, 0.10, 0.05),
        }
        grid_obj = self._perform_randomized_search(pipe, parameters)
        tuned_estimator = grid_obj.best_estimator_
        tuned_estimator.fit(self.X_train, self.y_train)
        return tuned_estimator

    def _perform_randomized_search(self, estimator, parameters):
        """
        Perform randomized search for hyperparameter tuning.

        Parameters:
            estimator: Classifier to tune.
            parameters (dict): Hyperparameters and their possible values.

        Returns:
            grid_obj (RandomizedSearchCV): Trained RandomizedSearchCV object.
        """
        grid_obj = RandomizedSearchCV(
            estimator,
            n_iter=self.n_iter_search,
            param_distributions=parameters,
            scoring=self.acc_scorer,
            cv=5,
            n_jobs=self.n_jobs,
        )
        grid_obj = grid_obj.fit(self.X_train, self.y_train)
        return grid_obj
