from typing import Optional

import click

from src.feature_engineering.etl import load_data
from src.feature_engineering.model_specific_transformations import (
    apply_one_hot_encoding,
    encode_categorical_features,
    log_transform_features,
    replace_values,
    save_datasets,
    split_data,
)
from utils.ml_logging import get_logger

# Set up logging
logger = get_logger()


@click.group()
def cli() -> None:
    """Execute before every command."""
    logger.info("Executing the pipeline component...")


@cli.command()
@click.option(
    "-i", "--input_path", type=str, required=True, help="Path to the data file"
)
@click.option(
    "-o",
    "--output_directory",
    type=str,
    required=True,
    help="Directory to save datasets",
)
@click.option(
    "-dt",
    "--date",
    type=str,
    required=False,
    help="Date to save datasets",
    default=None,
)
def run_feature_engineering(
    input_path: str, output_directory: str, date: Optional[str] = None
) -> None:
    """
    Run feature engineering process, save resulting datasets to specified directory.
    """
    logger.info(f"Running feature engineering with data from {input_path}")

    # Load and preprocess data
    df = load_data(input_path)
    transform_log_features = [
        "Credit_Limit",
        "Avg_Open_To_Buy",
        "Total_Amt_Chng_Q4_Q1",
        "Total_Trans_Amt",
        "Total_Ct_Chng_Q4_Q1",
        "Avg_Utilization_Ratio",
    ]
    df = log_transform_features(df, transform_log_features)

    transform_categorical_features = ["Gender", "Attrition_Flag"]
    df = encode_categorical_features(df, transform_categorical_features)

    transform_one_hot_encoding_features = ["Card_Category", "Marital_Status"]
    df = apply_one_hot_encoding(df, transform_one_hot_encoding_features)

    replace_struct = {
        "Attrition_Flag": {"Existing Customer": 0, "Attrited Customer": 1},
        "Education_Level": {
            "Doctorate": 5,
            "Post-Graduate": 4,
            "Graduate": 3,
            "College": 2,
            "High School": 1,
            "Unknown": 0,
            "Uneducated": -1,
        },
        "Income_Category": {
            "$120K +": 4,
            "$80K - $120K": 3,
            "$60K - $80K": 2,
            "$40K - $60K": 1,
            "Unknown": 0,
            "Less than $40K": -1,
        },
    }
    df = replace_values(df, replace_struct)

    # Split data and save datasets
    X_train, X_test, y_train, y_test = split_data(
        df, target_column="Attrition_Flag", test_size=0.30, random_state=1
    )
    save_datasets(X_train, X_test, y_train, y_test, output_directory, date)


if __name__ == "__main__":
    cli()
