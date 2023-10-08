import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn import metrics

from utils.logging_ml import get_logger

# Set up logging
logger = get_logger()


def make_confusion_matrix(model, X_test, y_actual, labels=[1, 0]):
    """
    Generate confusion matrix for the fitted model.

    Parameters:
    model (Model): Classifier to predict values of X_test.
    X_test (array): Test set.
    y_actual (array): Ground truth.
    labels (list): List of labels to create confusion matrix.

    Returns:
    None
    """
    y_predict = model.predict(X_test)
    cm = metrics.confusion_matrix(y_actual, y_predict, labels=labels)
    df_cm = pd.DataFrame(
        cm,
        index=[f"Actual - {label}" for label in labels],
        columns=[f"Predicted - {label}" for label in labels],
    )
    group_counts = ["{0:0.0f}".format(value) for value in cm.flatten()]
    group_percentages = ["{0:.2%}".format(value) for value in cm.flatten() / np.sum(cm)]
    labels = [f"{v1}\n{v2}" for v1, v2 in zip(group_counts, group_percentages)]
    labels = np.asarray(labels).reshape(2, 2)
    plt.figure(figsize=(10, 7))
    sns.heatmap(df_cm, annot=labels, fmt="")
    plt.ylabel("True label")
    plt.xlabel("Predicted label")


def get_metrics_score(model, X_train, y_train, X_test, y_test, log_scores=True):
    """
    Calculate different metric scores of the model - Accuracy, Recall, and Precision.

    Parameters:
    model (Model): Classifier to predict values of X.
    X_train (array): Training set.
    y_train (array): Training set labels.
    X_test (array): Test set.
    y_test (array): Test set labels.
    log_scores (bool): Flag to log the scores. Default is True.

    Returns:
    list: List containing metric scores.
    """
    # defining an empty list to store train and test results
    score_list = []

    # Predictions
    pred_train = model.predict(X_train)
    pred_test = model.predict(X_test)

    # Accuracy
    train_acc = model.score(X_train, y_train)
    test_acc = model.score(X_test, y_test)

    # Recall
    train_recall = metrics.recall_score(y_train, pred_train)
    test_recall = metrics.recall_score(y_test, pred_test)

    # Precision
    train_precision = metrics.precision_score(y_train, pred_train)
    test_precision = metrics.precision_score(y_test, pred_test)

    score_list.extend(
        (
            train_acc,
            test_acc,
            train_recall,
            test_recall,
            train_precision,
            test_precision,
        )
    )

    if log_scores:
        logger.info(f"Accuracy on training set : {train_acc}")
        logger.info(f"Accuracy on test set : {test_acc}")
        logger.info(f"Recall on training set : {train_recall}")
        logger.info(f"Recall on test set : {test_recall}")
        logger.info(f"Precision on training set : {train_precision}")
        logger.info(f"Precision on test set : {test_precision}")

    return score_list
