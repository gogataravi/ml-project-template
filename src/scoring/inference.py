import pickle
from typing import Any

from utils.ml_logging import get_logger

# Set up logging
logger = get_logger()


class ModelInference:
    """
    Class for running inference with saved models.

    Attributes:
        model_name (str): The name of the model to be loaded.
        model (Any): The loaded model.
        input_data (Any): Input data for prediction.
    """

    def __init__(self, model_name: str, input_data: Any):
        """
        Initialize with the model name and input data.

        Parameters:
            model_name (str): The name of the model to be loaded.
            input_data (Any): Input data for prediction.
        """
        self.model_name = model_name
        self.model = None
        self.input_data = input_data

    def load_model(self, model_path: str) -> None:
        """
        Load the model from a specified file using Pickle.

        Parameters:
            model_path (str): The path to the file where the model is saved.
        """
        try:
            logger.info(f"Loading model from {model_path}.")
            with open(model_path, "rb") as file:
                self.model = pickle.load(file)  # nosec
            logger.info(f"Model loaded successfully from {model_path}.")

        except Exception as e:
            logger.error(f"Error occurred while loading model from {model_path}: {e}")
            raise e

    def prepare_input_data(self) -> None:
        """
        Prepare the input data for prediction.
        Implement the necessary preprocessing steps that your data requires.
        """
        try:
            logger.info("Preparing input data for prediction.")
            # Add your data preparation logic here
            # ...
            logger.info("Input data prepared successfully.")

        except Exception as e:
            logger.error(f"Error occurred while preparing input data: {e}")
            raise e

    def run_inference(self) -> Any:
        """
        Run model inference on the prepared input data.

        Returns:
            The model's predictions on the input data.
        """
        try:
            if not self.model:
                logger.error("Model not loaded. Cannot run inference.")
                return None

            logger.info(f"Running inference with model {self.model_name}.")
            predictions = self.model.predict(self.input_data)
            logger.info("Inference completed successfully.")
            return predictions

        except Exception as e:
            logger.error(
                f"Error occurred during inference with model {self.model_name}: {e}"
            )
            raise e

    def save_predictions(self, predictions: Any, output_path: str) -> None:
        """
        Save the predictions to a specified location.

        Parameters:
            predictions (Any): The predictions to be saved.
            output_path (str): The path where predictions will be saved.
        """
        try:
            logger.info(f"Saving predictions to {output_path}.")
            # Implement your preferred method of saving predictions here
            # For example, if predictions is a numpy array and you want to save it as a CSV:
            # np.savetxt(output_path, predictions, delimiter=',')
            # Or if working with pandas:
            # predictions.to_csv(output_path, index=False)
            logger.info(f"Predictions saved successfully to {output_path}.")

        except Exception as e:
            logger.error(
                f"Error occurred while saving predictions to {output_path}: {e}"
            )
            raise e
