import pandas as pd
import joblib
import os
import yaml
import logging

from features.featureselector import FeatureSelector
from features.Outlierhandler import OutlierHandler
from features.featuremapping import ColumnRenamer
from src.datapreprocessing import DataProcessor

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[
        logging.FileHandler("pipeline.log"),
        logging.StreamHandler()
    ]
)

def load_config():
    """Load the configuration file for file paths."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, 'config', 'config.yml')

    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        logging.error(f"Failed to load config from {config_path}: {e}")
        raise

def load_model(model_path):
    """Load the trained model from a pickle file."""
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    with open(model_path, 'rb') as f:
        return joblib.load(f)

def save_predictions(predictions, output_path):
    """Save predictions to a CSV file."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    pd.DataFrame(predictions, columns=["Prediction"]).to_csv(output_path, index=False)

def main():
    logging.info("Pipeline started.")
    try:
        config = load_config()
        logging.info("Configuration loaded.")
    except Exception as e:
        logging.error(f"Error loading configuration: {e}")
        return

    try:
        logging.info("Starting data processing...")
        raw_file_path = config['build']['rawdata']
        processed_file_path = config['build']['cleaned_data_path']

        processor = DataProcessor(raw_file_path, processed_file_path)
        processor.load_data()
        processor.remove_high_null_columns(threshold=70)
        processor.impute_numerical_values()
        processor.impute_categorical_values()
        columns_with_nulls = processor.find_columns_with_nulls()
        processor.impute_with_iterative(columns_with_nulls)
        processor.save_cleaned_data()
        logging.info("Data processing completed.")
    except Exception as e:
        logging.error(f"Data processing error: {e}")
        return

    try:
        logging.info("Starting feature selection...")
        cleaned_data_path = config['build']['cleaned_data_path']
        feature_output_path = config['build']['feature_output_path']

        selector = FeatureSelector(input_path=cleaned_data_path, target_col='phq8_total', top_k=10)
        selector.load_data()
        selector.correlation_method()
        selector.kbest_method()
        selector.random_forest_method()
        logging.info("Feature selection completed.")
    except Exception as e:
        logging.error(f"Feature selection error: {e}")
        return

    try:
        logging.info("Starting column renaming...")
        renamed_output_path = config['build']['randomforestor_data']
        renamer = ColumnRenamer(input_path=feature_output_path, output_path=renamed_output_path)
        renamer.load_data()
        renamer.rename_columns()
        renamed_columns = renamer.get_renamed_features()
        renamer.save_data()
        logging.info(f"Renamed columns: {renamed_columns}")
        logging.info("Column renaming completed.")
    except Exception as e:
        logging.error(f"Column renaming error: {e}")
        return

    try:
        logging.info("Starting outlier handling...")
        xgboost_output_path = config['build']['xgboostdata']
        outlier_handler = OutlierHandler(input_path=renamed_output_path, output_path=xgboost_output_path)
        outlier_handler.load_data()
        outlier_handler.handle_outliers()
        outlier_handler.save_data()
        logging.info("Outlier handling completed.")
    except Exception as e:
        logging.error(f"Outlier handling error: {e}")
        return

    try:
        logging.info("Starting prediction...")
        model_path = config['build']['model_output_path']
        prediction_output_path = config['build']['prediction_output_path']

        model = load_model(model_path)
        logging.info(f"Model loaded from {model_path}.")

        predict_data = pd.read_csv(renamed_output_path, index_col=False)
        predict_data = predict_data[renamed_columns]
        logging.info(f"Prediction data shape: {predict_data.shape}")

        predictions = model.predict(predict_data)
        save_predictions(predictions, prediction_output_path)
        logging.info(f"Predictions saved to {prediction_output_path}.")
    except Exception as e:
        logging.error(f"Prediction error: {e}")
        return

    logging.info("Pipeline executed successfully.")

if __name__ == "__main__":
    main()
    