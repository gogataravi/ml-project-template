import pandas as pd
import numpy as np
from typing import Tuple
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from utils.logging import log_function_call
from utils.logging import get_logger

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


def split_data(df: pd.DataFrame, target_column: str, test_size: float, random_state: int) -> tuple:
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
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state, stratify=y)
    return X_train, X_test, y_train, y_test


def perform_upsampling(X_train: pd.DataFrame, y_train: pd.Series, strategy: float = 1, k_neighbors: int = 5, random_state: int = 1) -> Tuple[pd.DataFrame, pd.Series]:
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

    sm = SMOTE(sampling_strategy=strategy, k_neighbors=k_neighbors, random_state=random_state)
    X_train_res, y_train_res = sm.fit_resample(X_train, y_train.ravel())

    logger.info(f"After Upsampling, counts of label '1': {sum(y_train_res==1)}")
    logger.info(f"After Upsampling, counts of label '0': {sum(y_train_res==0)}")

    return X_train_res, y_train_res


def perform_downsampling(X_train: pd.DataFrame, y_train: pd.Series) -> Tuple[pd.DataFrame, pd.Series]:
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
