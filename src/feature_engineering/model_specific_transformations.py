import os
from datetime import datetime
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from sklearn import preprocessing
from sklearn.model_selection import train_test_split

from utils.ml_logging import get_logger, log_function_call

# Set up logging
logger = get_logger()


@log_function_call("Feature Engineering")
def log_transform_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Apply log transformation to specified features in the DataFrame.

    :param df: Input DataFrame.
    :param features: List of feature names to be log-transformed.
    :return: DataFrame with log-transformed features.
    """
    for feature in features:
        df[feature] = df[feature].apply(lambda x: np.log(x + 1))
    return df


@log_function_call("Feature Engineering")
def encode_categorical_features(df: pd.DataFrame, features: list) -> pd.DataFrame:
    """
    Encode categorical features using LabelEncoder.

    :param df: Input DataFrame.
    :param features: List of categorical feature names to be encoded.
    :return: DataFrame with encoded categorical features.
    """
    labelencoder = preprocessing.LabelEncoder()
    for feature in features:
        df[feature] = labelencoder.fit_transform(df[feature])
    return df


@log_function_call("Feature Engineering")
def apply_one_hot_encoding(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Apply one-hot encoding to specified columns in the DataFrame.

    :param df: Input DataFrame.
    :param columns: List of columns names to be one-hot encoded.
    :return: DataFrame with one-hot encoded columns.
    """
    return pd.get_dummies(df, columns=columns)


@log_function_call("Feature Engineering")
def replace_values(df: pd.DataFrame, replace_structure: dict) -> pd.DataFrame:
    """
    Replace values in the DataFrame based on a provided structure.

    :param df: Input DataFrame.
    :param replace_structure: Dictionary with structure {column: {old_value: new_value}}.
    :return: DataFrame with values replaced as per replace_structure.
    """
    return df.replace(replace_structure)


@log_function_call("Feature Engineering")
def split_data(
    df: pd.DataFrame, target_column: str, test_size: float, random_state: int
) -> tuple:
    """
    Split data into training and testing sets and separate target variable.

    :param df: Input DataFrame.
    :param target_column: The name of the target variable column.
    :param test_size: Proportion of the dataset included in the test split.
    :param random_state: Seed used by the random number generator for shuffling.
    :return: Tuple of (X_train, X_test, y_train, y_test).
    """
    X = df.drop(target_column, axis=1)
    y = df.pop(target_column)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


@log_function_call("Feature Engineering")
def perform_upsampling(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    strategy: float = 1,
    k_neighbors: int = 5,
    random_state: int = 1,
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Perform upsampling on the dataset to balance it.

    Parameters:
    X_train (DataFrame): The input features.
    y_train (Series): The target variable.
    strategy (float): The sampling strategy for SMOTE. Default is 1.
    k_neighbors (int): Number of nearest neighbours to used to construct synthetic samples. Default is 5.
    random_state (int): The seed used by the random number generator. Default is 1.

    Returns:
    X_train_res (DataFrame): The input features after resampling.
    y_train_res (Series): The target variable after resampling.
    """
    logger.info(f"Before Upsampling, counts of label '1': {sum(y_train==1)}")
    logger.info(f"Before Upsampling, counts of label '0': {sum(y_train==0)}")

    sm = SMOTE(
        sampling_strategy=strategy, k_neighbors=k_neighbors, random_state=random_state
    )
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())

    logger.info(f"After Upsampling, counts of label '1': {sum(y_train_res==1)}")
    logger.info(f"After Upsampling, counts of label '0': {sum(y_train_res==0)}")

    return X_train_res, y_train_res


@log_function_call("Feature Engineering")
def perform_downsampling(
    X_train: pd.DataFrame, y_train: pd.Series
) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Perform downsampling on the dataset to balance it.

    Parameters:
    X_train (DataFrame): The input features.
    y_train (Series): The target variable.

    Returns:
    X_train_res (DataFrame): The input features after resampling.
    y_train_res (Series): The target variable after resampling.
    """
    logger.info(f"Before Downsampling, counts of label '1': {sum(y_train==1)}")
    logger.info(f"Before Downsampling, counts of label '0': {sum(y_train==0)}")

    rus = RandomUnderSampler()
    X_train_res, y_train_res = rus.fit_resample(X_train, y_train)

    logger.info(f"After Downsampling, counts of label '1': {sum(y_train_res==1)}")
    logger.info(f"After Downsampling, counts of label '0': {sum(y_train_res==0)}")

    return X_train_res, y_train_res


def save_datasets(
    X_train: Union[pd.DataFrame, pd.Series],
    X_test: Union[pd.DataFrame, pd.Series],
    y_train: Union[pd.DataFrame, pd.Series],
    y_test: Union[pd.DataFrame, pd.Series],
    directory: str,
    date: Optional[str] = None,
) -> None:
    """
    Save X_train, X_test, y_train, and y_test as CSV files.

    Parameters:
    X_train (DataFrame/Series): Training data features.
    X_test (DataFrame/Series): Test data features.
    y_train (DataFrame/Series): Training data target.
    y_test (DataFrame/Series): Test data target.
    directory (str): Directory to save the CSV files.
    date (str, optional): Date string. Defaults to today's date in format 'day_month_year'.
    """

    # Set default date to today if not provided
    if date is None:
        date = datetime.today().strftime("%d_%m_%Y")

    # Create directory if it does not exist
    if not os.path.exists(directory):
        os.makedirs(directory)
        logger.info(f"Created directory: {directory}")

    # Convert data to DataFrame if they are Series
    if isinstance(X_train, pd.Series):
        X_train = X_train.to_frame()
    if isinstance(X_test, pd.Series):
        X_test = X_test.to_frame()
    if isinstance(y_train, pd.Series):
        y_train = y_train.to_frame()
    if isinstance(y_test, pd.Series):
        y_test = y_test.to_frame()

    # Save datasets as CSV files
    X_train.to_csv(os.path.join(directory, f"X_train_{date}.csv"), index=False)
    logger.info(f"X_train saved at {os.path.join(directory, f'X_train_{date}.csv')}")

    X_test.to_csv(os.path.join(directory, f"X_test_{date}.csv"), index=False)
    logger.info(f"X_test saved at {os.path.join(directory, f'X_test_{date}.csv')}")

    y_train.to_csv(os.path.join(directory, f"y_train_{date}.csv"), index=False)
    logger.info(f"y_train saved at {os.path.join(directory, f'y_train_{date}.csv')}")

    y_test.to_csv(os.path.join(directory, f"y_test_{date}.csv"), index=False)
    logger.info(f"y_test saved at {os.path.join(directory, f'y_test_{date}.csv')}")


def load_datasets(
    directory: str, date: Optional[str] = None
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Load X_train, X_test, y_train, and y_test from CSV files.

    Parameters:
    directory (str): Directory to load the CSV files from.
    date (str, optional): Date string. Defaults to today's date in format 'day_month_year'.

    Returns:
    tuple: Tuple containing DataFrames (X_train, X_test, y_train, y_test).
    """

    # Set default date to today if not provided
    if date is None:
        date = datetime.today().strftime("%d_%m_%Y")

    # Define file paths
    x_train_path = os.path.join(directory, f"X_train_{date}.csv")
    x_test_path = os.path.join(directory, f"X_test_{date}.csv")
    y_train_path = os.path.join(directory, f"y_train_{date}.csv")
    y_test_path = os.path.join(directory, f"y_test_{date}.csv")

    # Check if files exist
    if not all(
        os.path.exists(path)
        for path in [x_train_path, x_test_path, y_train_path, y_test_path]
    ):
        logger.error(
            f"One or more files not found in directory: {directory} for date: {date}"
        )
        raise FileNotFoundError("One or more dataset files not found.")

    # Load datasets
    X_train = pd.read_csv(x_train_path)
    X_test = pd.read_csv(x_test_path)
    y_train = pd.read_csv(y_train_path)
    y_test = pd.read_csv(y_test_path)

    logger.info(f"Datasets loaded from directory: {directory} for date: {date}")

    return X_train, X_test, y_train, y_test
