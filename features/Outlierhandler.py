import os
import pandas as pd
import numpy as np
from scipy.stats import zscore
import yaml

class OutlierHandler:
    def __init__(self, input_path, output_path, z_thresh=3.0):
        """
        Initialize OutlierHandler with paths for input/output and Z-score threshold.
        
        :param input_path: Path to input data file (CSV)
        :param output_path: Path to save the cleaned data
        :param z_thresh: Z-score threshold to cap outliers (default is 3.0)
        """
        self.input_path = input_path
        self.output_path = output_path
        self.z_thresh = z_thresh
        self.df = None

    def load_data(self):
        """Load the input data into a DataFrame."""
        try:
            self.df = pd.read_csv(self.input_path)
            print("Data loaded successfully.")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def handle_outliers(self):
        """Detect and handle outliers by capping them using Z-score."""
        try:
            # Select numeric columns
            X_numeric = self.df.select_dtypes(include=[np.number])
            
            for col in X_numeric.columns:
                # Ignore columns with all NaN values
                if X_numeric[col].isnull().all():
                    continue
                
                # Calculate Z-scores
                col_zscore = zscore(X_numeric[col].dropna())
                
                # Calculate mean and std for capping
                mean = X_numeric[col].mean()
                std = X_numeric[col].std()
                
                # Define capping limits using Z-score
                lower_bound = mean - self.z_thresh * std
                upper_bound = mean + self.z_thresh * std
                
                # Cap outliers within the calculated range
                self.df[col] = self.df[col].clip(lower=lower_bound, upper=upper_bound)
            
            print("Outliers handled using Z-score capping.")
        except Exception as e:
            print(f"Error during outlier handling: {e}")
            raise

    def save_data(self):
        """Save the cleaned data to the output path."""
        try:
            self.df.to_csv(self.output_path, index=False)
            print("Cleaned data saved successfully.")
        except Exception as e:
            print(f"Error saving cleaned data: {e}")
            raise

def load_config():
    """Load the configuration file for file paths."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '..', 'config', 'config.yml')

    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            return config_data
    except Exception as e:
        print(f"Failed to load config from {config_path}: {e}")
        raise

if __name__ == "__main__":
    # Load config to get paths
    config = load_config()

    # Paths from config
    cleaned_data_path = config['build']['feature_output_path']
    xgboost_output_path = config['build']['xgboostdata']

    # Initialize the OutlierHandler class
    outlier_handler = OutlierHandler(input_path=cleaned_data_path, output_path=xgboost_output_path)
    
    # Load data
    outlier_handler.load_data()
    
    # Handle outliers
    outlier_handler.handle_outliers()
    
    # Save the cleaned data
    outlier_handler.save_data()
