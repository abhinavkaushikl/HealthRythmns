import os
import pandas as pd
import numpy as np
import yaml
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split

class FeatureSelector:
    def __init__(self, input_path, target_col='phq8_total', top_k=10):
        self.input_path = input_path
        self.target_col = target_col
        self.top_k = top_k
        self.df = None
        self.X = None
        self.y = None
        self.X_numeric = None
        self.selected_features = []

    def load_data(self):
        try:
            self.df = pd.read_csv(self.input_path)
            self.X = self.df.drop(columns=[self.target_col])
            self.y = self.df[self.target_col]
            self.X_numeric = self.X.select_dtypes(include=[np.number])
        except Exception as e:
            print(f"Error loading cleaned data: {e}")
            raise

    def correlation_method(self):
        try:
            corr = self.X_numeric.corrwith(self.y).abs().sort_values(ascending=False)
            top_corr = corr.head(self.top_k)
            self.selected_features.extend(top_corr.index.tolist())
        except Exception as e:
            print(f"Correlation method failed: {e}")

    def kbest_method(self):
        try:
            selector = SelectKBest(score_func=f_regression, k=self.top_k)
            selector.fit(self.X_numeric, self.y)
            scores = pd.Series(selector.scores_, index=self.X_numeric.columns)
            top_kbest = scores.sort_values(ascending=False).head(self.top_k)
            self.selected_features.extend(top_kbest.index.tolist())
        except Exception as e:
            print(f"SelectKBest method failed: {e}")

    def random_forest_method(self):
        try:
            X_train, _, y_train, _ = train_test_split(self.X_numeric, self.y, test_size=0.2, random_state=42)
            model = RandomForestRegressor(random_state=42)
            model.fit(X_train, y_train)
            importances = pd.Series(model.feature_importances_, index=self.X_numeric.columns)
            top_rf = importances.sort_values(ascending=False).head(self.top_k)
            self.selected_features.extend(top_rf.index.tolist())
        except Exception as e:
            print(f"Random Forest method failed: {e}")

    def get_unique_features(self):
        return list(set(self.selected_features))

    def save_selected_features(self, output_path):
        try:
            features = self.get_unique_features()
            selected_data = self.df[features + [self.target_col]]
            selected_data.to_csv(output_path, index=False)
        except Exception as e:
            print(f"Error saving selected features: {e}")




