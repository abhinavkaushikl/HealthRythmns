import pandas as pd
import numpy as np
from sklearn.impute import SimpleImputer
from sklearn.experimental import enable_iterative_imputer  # noqa
from sklearn.impute import IterativeImputer
import os
import yaml
import joblib 
import pickle

class DataProcessor:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None
        self.df_cleaned = None

    def load_data(self):
        try:
            self.df = pd.read_csv(self.input_path)
            print("Data loaded successfully.")
        except Exception as e:
            print(f"Error loading data: {e}")

    def remove_high_null_columns(self, threshold=70):
        try:
            null_percentage = self.df.isnull().mean() * 100
            cols_to_remove = null_percentage[null_percentage > threshold].index
            self.df_cleaned = self.df.drop(columns=cols_to_remove)
            print(f"Removed columns with more than {threshold}% missing values: {cols_to_remove.tolist()}")
        except Exception as e:
            print(f"Error in removing high null columns: {e}")

    def impute_numerical_values(self):
        try:
            numerical_cols = self.df_cleaned.select_dtypes(include=['number']).columns
            skewness = self.df_cleaned[numerical_cols].skew()

            symmetric_columns = skewness[(skewness >= -0.5) & (skewness <= 0.5)].index.tolist()
            skewed_columns = skewness[skewness.abs() > 1].index.tolist()

            print(f"Symmetric columns: {symmetric_columns}")
            print(f"Skewed columns: {skewed_columns}")

            imputer_mean = SimpleImputer(strategy='mean')
            self.df_cleaned[symmetric_columns] = imputer_mean.fit_transform(self.df_cleaned[symmetric_columns])

            imputer_median = SimpleImputer(strategy='median')
            self.df_cleaned[skewed_columns] = imputer_median.fit_transform(self.df_cleaned[skewed_columns])

            print("Numerical imputation complete.")

        except Exception as e:
            print(f"Error in numerical imputation: {e}")

    def impute_categorical_values(self):
        try:
            if 'ethnicity' in self.df_cleaned.columns:
                self.df_cleaned['ethnicity'] = self.df_cleaned['ethnicity'].fillna(self.df_cleaned['ethnicity'].mode()[0])
                print("Categorical column 'ethnicity' imputed with mode.")
        except Exception as e:
            print(f"Error in categorical imputation: {e}")

    def find_columns_with_nulls(self):
        try:
            null_percentage = self.df_cleaned.isnull().mean() * 100
            return null_percentage[null_percentage > 0].index.tolist()
        except Exception as e:
            print(f"Error finding columns with nulls: {e}")
            return []

    def impute_with_iterative(self, columns):
        try:
            imputer = IterativeImputer(random_state=42)
            self.df_cleaned[columns] = imputer.fit_transform(self.df_cleaned[columns])
            print("Iterative imputation completed.")
        except Exception as e:
            print(f"Error in iterative imputation: {e}")

    def save_cleaned_data(self):
        try:
            self.df_cleaned.to_csv(self.output_path, index=False)
            print("Cleaned data saved successfully.")
        except Exception as e:
            print(f"Error saving cleaned data: {e}")
            
# Updated config loader to use 'config/config.yml'
def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '..', 'config', 'config.yml')

    print(f"Looking for config at: {config_path}")

    try:
        with open(config_path, 'r') as f:
            config_data = yaml.safe_load(f)
            if config_data is None:
                raise ValueError("YAML file loaded as None — check if it's empty or malformed.")
            print("Loaded config:", config_data)
            return config_data
    except Exception as e:
        print(f"Failed to load config: {e}")
        raise

if __name__ == "__main__":
    config = load_config()
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
