import json
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import yaml

class RandomForestTrainer:
    def __init__(self, data_path, best_params_path, metrics_output_path):
        self.data_path = data_path
        self.best_params_path = best_params_path
        self.metrics_output_path = metrics_output_path
        #self.model_output_path = model_output_path
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = (None,) * 4

    def load_data(self):
        df = pd.read_csv(self.data_path)
        X = df.drop(columns=['Depression'])
        y = df['Depression']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print("Data loaded and split.")

    def load_best_params(self):
        if not os.path.exists(self.best_params_path) or os.path.getsize(self.best_params_path) == 0:
            print("Best parameters file is missing or empty.")
            return {}

        with open(self.best_params_path, 'r') as f:
            try:
                best_params = json.load(f)
            except json.JSONDecodeError:
                print("Best parameters file is not valid JSON.")
                return {}
        print(f"Best hyperparameters loaded: {best_params}")
        return best_params

    def train_model(self, best_params):
        self.model = RandomForestRegressor(**best_params, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        print("Model trained with best hyperparameters.")

    def evaluate_and_save_metrics(self):
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        metrics = {
            'train': {
                'r2': r2_score(self.y_train, y_train_pred),
                'rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred))
            },
            'test': {
                'r2': r2_score(self.y_test, y_test_pred),
                'rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred))
            }
        }

        os.makedirs(os.path.dirname(self.metrics_output_path), exist_ok=True)
        pd.DataFrame(metrics).to_csv(self.metrics_output_path)
        print(f"Metrics saved to {self.metrics_output_path}")

    

def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '..', 'config', 'config.yml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load config: {e}")
        raise

if __name__ == "__main__":
    config = load_config()

    trainer = RandomForestTrainer(
        data_path=config['build']['randomforestor_data'],
        best_params_path=config['build']['best_params_path'],
        metrics_output_path=config['build']['metrics_output_path'],
      
    )

    trainer.load_data()
    best_params = trainer.load_best_params()
    trainer.train_model(best_params)
    trainer.evaluate_and_save_metrics()
# train.py
import json
import pandas as pd
import numpy as np
import os
import joblib
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error
import yaml

class RandomForestTrainer:
    def __init__(self, data_path, best_params_path, metrics_output_path, model_output_path):
        self.data_path = data_path
        self.best_params_path = best_params_path
        self.metrics_output_path = metrics_output_path
        self.model_output_path = model_output_path
        self.model = None
        self.X_train, self.X_test, self.y_train, self.y_test = (None,) * 4

    def load_data(self):
        df = pd.read_csv(self.data_path, index_col=False)
        X = df.drop(columns=['Depression'])
        y = df['Depression']
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        print("Data loaded and split.")

    def load_best_params(self):
        if not os.path.exists(self.best_params_path) or os.path.getsize(self.best_params_path) == 0:
            print("Best parameters file is missing or empty.")
            return {}

        with open(self.best_params_path, 'r') as f:
            try:
                best_params = json.load(f)
            except json.JSONDecodeError:
                print("Best parameters file is not valid JSON.")
                return {}
        print(f"Best hyperparameters loaded: {best_params}")
        return best_params

    def train_model(self, best_params):
        self.model = RandomForestRegressor(**best_params, random_state=42)
        self.model.fit(self.X_train, self.y_train)
        print("Model trained with best hyperparameters.")

    def evaluate_and_save_metrics(self):
        y_train_pred = self.model.predict(self.X_train)
        y_test_pred = self.model.predict(self.X_test)

        metrics = {
            'train': {
                'r2': r2_score(self.y_train, y_train_pred),
                'rmse': np.sqrt(mean_squared_error(self.y_train, y_train_pred))
            },
            'test': {
                'r2': r2_score(self.y_test, y_test_pred),
                'rmse': np.sqrt(mean_squared_error(self.y_test, y_test_pred))
            }
        }

        os.makedirs(os.path.dirname(self.metrics_output_path), exist_ok=True)
        pd.DataFrame(metrics).to_csv(self.metrics_output_path)
        print(f"Metrics saved to {self.metrics_output_path}")

    def save_model(self):
        os.makedirs(os.path.dirname(self.model_output_path), exist_ok=True)
        joblib.dump(self.model, self.model_output_path)
        print(f"Model saved to {self.model_output_path}")

def load_config():
    script_dir = os.path.dirname(os.path.abspath(__file__))
    config_path = os.path.join(script_dir, '..', 'config', 'config.yml')
    try:
        with open(config_path, 'r') as f:
            return yaml.safe_load(f)
    except Exception as e:
        print(f"Failed to load config: {e}")
        raise

if __name__ == "__main__":
    config = load_config()

    trainer = RandomForestTrainer(
        data_path=config['build']['randomforestor_data'],
        best_params_path=config['build']['best_params_path'],
        metrics_output_path=config['build']['metrics_output_path'],
        model_output_path=config['build']['model_output_path']
    )

    trainer.load_data()
    best_params = trainer.load_best_params()
    trainer.train_model(best_params)
    trainer.evaluate_and_save_metrics()
    trainer.save_model()
