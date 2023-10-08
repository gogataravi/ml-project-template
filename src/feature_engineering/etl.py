import pandas as pd
from utils.logging import get_logger
from utils.logging import log_function_call

# Set up logging
logger = get_logger()

@log_function_call("etl")
def load_data(file_path: str) -> pd.DataFrame:
    """
    Load data from a CSV file, report missing values, and drop the 'CLIENTNUM' column.

    :param file_path: The path to the CSV file.
    :return: The loaded and processed DataFrame.
    :raises FileNotFoundError: If the file at file_path does not exist.
    :raises ValueError: If any column has more than 20 percent missing values.
    """
    try:
        df = pd.read_csv(file_path, na_values=["n.a.", "?", "NA", "n/a", "na", "--", "nan"])
        df = df.drop(columns=['CLIENTNUM'])

        # Validate missing data
        validate_missing_data(df, threshold=20.0)
        
        return df
    except FileNotFoundError:
        logger.error(f"The file at path {file_path} does not exist.")
        raise

@log_function_call("etl")
def validate_missing_data(df: pd.DataFrame, threshold: float):
    """
    Validate if any column in the DataFrame has missing values more than the specified threshold.

    :param df: Input DataFrame to be validated.
    :param threshold: The percentage threshold for missing values.
    :raises ValueError: If any column has missing values more than the specified threshold.
    """
    percent_missing = df.isnull().sum() * 100 / len(df)
    missing_value_df = pd.DataFrame({
            'column_name': df.columns,
            'percent_missing': percent_missing
        })
        
    logger.info(missing_value_df[['column_name', 'percent_missing']])
    if any(percent_missing > threshold):
        error_columns = percent_missing[percent_missing > threshold].index.tolist()
        logger.error(f"Columns {error_columns} have more than {threshold}% missing values.")
        raise ValueError(f"Columns {error_columns} have more than {threshold}% missing values.")


