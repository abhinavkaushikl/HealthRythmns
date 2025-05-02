import os
import pandas as pd
import numpy as np
import yaml
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class ColumnRenamer:
    def __init__(self, input_path, output_path):
        self.input_path = input_path
        self.output_path = output_path
        self.df = None
        self.training_features = []

        self.rename_dict = {
            'phq_q1': 'Interest',
            'phq_q2': 'Mood',
            'phq_q3': 'Sleep',
            'phq_q4': 'Energy',
            'phq_q5': 'Appetite',
            'phq_q6': 'Worthlessness',
            'phq_q7': 'Concentration',
            'phq_q8': 'Psychomotor',
            'phq_q9': 'Suicidality',
            'phq2_total': 'Screener',
            'phq8_total': 'Depression',
            'phq9_total': 'Severity',
            'age_num': 'Age',
            'age_bucket': 'AgeGroup',
            'total_activity_duration_s14': 'Activity'
        }

    def load_data(self):
        try:
            self.df = pd.read_csv(self.input_path)
            print(f"Data loaded from {self.input_path}.")
        except Exception as e:
            print(f"Error loading data: {e}")
            raise

    def rename_columns(self):
        try:
            self.df.rename(columns=self.rename_dict, inplace=True)
            print("Columns renamed successfully.")
        except Exception as e:
            print(f"Column renaming failed: {e}")
            raise

    def save_data(self):
        try:
            os.makedirs(os.path.dirname(self.output_path), exist_ok=True)
            self.df.to_csv(self.output_path, index=False)
            print(f"Renamed DataFrame saved to {self.output_path}.")
        except Exception as e:
            print(f"Error saving renamed data: {e}")
            raise
        
    def get_renamed_features(self):
        """Return the columns of the DataFrame after renaming, excluding 'Depression'."""
        columns = list(self.df.columns)
        if 'Depression' in columns:
            columns.remove('Depression')
        return list(columns)
    


