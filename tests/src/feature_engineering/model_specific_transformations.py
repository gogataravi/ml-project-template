from unittest.mock import MagicMock, Mock, patch

import numpy as np
import pandas as pd

from src.feature_engineering.model_specific_transformations import (
    apply_one_hot_encoding,
    encode_categorical_features,
    log_transform_features,
    perform_upsampling,
    replace_values,
    split_data,
)


def test_log_transform_features():
    # Arrange
    test_data = {
        "Credit_Limit": [1, 2, 3, 4],
        "Avg_Open_To_Buy": [5, 6, 7, 8],
        "Total_Amt_Chng_Q4_Q1": [9, 10, 11, 12],
        "Total_Trans_Amt": [100, 200, 300, 400],
        "Total_Ct_Chng_Q4_Q1": [0, 0, 0, 0],
        "Avg_Utilization_Ratio": [0.1, 0.2, 0.3, 0.4],
    }
    features = [
        "Credit_Limit",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
    ]
    expected_output = {
        "Credit_Limit": [np.log(2), np.log(3), np.log(4), np.log(5)],
        "Avg_Open_To_Buy": [np.log(6), np.log(7), np.log(8), np.log(9)],
        "Total_Amt_Chng_Q4_Q1": [np.log(10), np.log(11), np.log(12), np.log(13)],
        "Total_Trans_Amt": [np.log(101), np.log(201), np.log(301), np.log(401)],
        "Total_Ct_Chng_Q4_Q1": [0, 0, 0, 0],  # Log(1) = 0
        "Avg_Utilization_Ratio": [np.log(1.1), np.log(1.2), np.log(1.3), np.log(1.4)],
    }
    test_df = pd.DataFrame(test_data)

    # Act
    result_df = log_transform_features(test_df, features)

    # Assert
    pd.testing.assert_frame_equal(
        result_df, pd.DataFrame(expected_output), check_dtype=False
    )


def test_encode_categorical_features():
    # Arrange
    test_data = {
        "Attrition_Flag": [
            "Existing Customer",
            "Attrited Customer",
            "Existing Customer",
            "Attrited Customer",
        ],
        "Gender": ["M", "F", "F", "M"],
    }
    features = ["Attrition_Flag", "Gender"]
    expected_output = {
        "Attrition_Flag": [1, 0, 1, 0],  # Encoded values
        "Gender": [1, 0, 0, 1],  # Encoded values
    }
    test_df = pd.DataFrame(test_data)

    # Act
    result_df = encode_categorical_features(test_df, features)

    # Assert
    pd.testing.assert_frame_equal(
        result_df, pd.DataFrame(expected_output), check_dtype=False
    )


def test_apply_one_hot_encoding():
    # Arrange
    test_data = {
        "Card_Category": ["Blue", "Silver", "Gold", "Platinum"],
        "Marital_Status": ["Married", "Single", "Divorced", "Unknown"],
    }
    one_hot_columns = ["Card_Category", "Marital_Status"]
    test_df = pd.DataFrame(test_data)

    expected_output_data = {
        "Card_Category_Blue": [1, 0, 0, 0],
        "Card_Category_Silver": [0, 1, 0, 0],
        "Card_Category_Gold": [0, 0, 1, 0],
        "Card_Category_Platinum": [0, 0, 0, 1],
        "Marital_Status_Married": [1, 0, 0, 0],
        "Marital_Status_Single": [0, 1, 0, 0],
        "Marital_Status_Divorced": [0, 0, 1, 0],
        "Marital_Status_Unknown": [0, 0, 0, 1],
    }
    expected_output_df = pd.DataFrame(expected_output_data)

    # Act
    result_df = apply_one_hot_encoding(test_df, one_hot_columns)

    # Sort columns of both DataFrames for consistent ordering
    result_df = result_df.sort_index(axis=1)
    expected_output_df = expected_output_df.sort_index(axis=1)

    # Assert
    pd.testing.assert_frame_equal(result_df, expected_output_df, check_dtype=False)


def test_replace_values():
    # Arrange
    test_data = {
        "Attrition_Flag": [
            "Existing Customer",
            "Attrited Customer",
            "Existing Customer",
        ],
        "Education_Level": ["Doctorate", "Unknown", "Graduate"],
        "Income_Category": ["$120K +", "Unknown", "Less than $40K"],
    }
    replace_structure = {
        "Attrition_Flag": {"Existing Customer": 0, "Attrited Customer": 1},
        "Education_Level": {"Doctorate": 5, "Unknown": 0, "Graduate": 3},
        "Income_Category": {"$120K +": 4, "Unknown": 0, "Less than $40K": -1},
    }
    test_df = pd.DataFrame(test_data)
    expected_output_data = {
        "Attrition_Flag": [0, 1, 0],
        "Education_Level": [5, 0, 3],
        "Income_Category": [4, 0, -1],
    }
    expected_output_df = pd.DataFrame(expected_output_data)

    # Act
    result_df = replace_values(test_df, replace_structure)

    # Assert
    pd.testing.assert_frame_equal(result_df, expected_output_df, check_dtype=False)


def test_split_data():
    # Arrange
    test_data = {
        "Attrition_Flag": [0, 1, 0, 1, 0],
        "Feature_1": [1, 2, 3, 4, 5],
        "Feature_2": [5, 4, 3, 2, 1],
    }
    test_df = pd.DataFrame(test_data)
    target_column = "Attrition_Flag"
    test_size = 0.4
    random_state = 1

    # Act
    X_train, X_test, y_train, y_test = split_data(
        test_df.copy(), target_column, test_size, random_state
    )

    # Assert
    assert len(X_train) == 3  # 60% of data for training
    assert len(X_test) == 2  # 40% of data for testing
    assert set(y_train) == {0, 1}  # Both classes present in y_train
    assert set(y_test) == {0, 1}  # Both classes present in y_test


# Sample test data
X_train = pd.DataFrame(np.random.rand(10, 5), columns=list("ABCDE"))
y_train = pd.Series(np.random.choice([0, 1], size=(10,)))

# Mock objects for the tests
mock_smote_instance = Mock()
mock_smote_instance.fit_resample = MagicMock(return_value=(X_train, y_train))

mock_rus_instance = Mock()
mock_rus_instance.fit_resample = MagicMock(return_value=(X_train, y_train))


@patch(
    "src.feature_engineering.model_specific_transformations.SMOTE",
    return_value=mock_smote_instance,
)
def test_perform_upsampling(mock_smote_class):
    # Call the function
    X_res, y_res = perform_upsampling(X_train, y_train)

    # Assert that SMOTE was instantiated
    mock_smote_class.assert_called_once_with(
        sampling_strategy=1, k_neighbors=5, random_state=1
    )

    # If fit_resample expected to return arrays, use proper assertion for arrays
    np.testing.assert_array_equal(
        mock_smote_instance.fit_resample.call_args[0][0], X_train.to_numpy()
    )
    np.testing.assert_array_equal(
        mock_smote_instance.fit_resample.call_args[0][1], y_train.to_numpy()
    )
