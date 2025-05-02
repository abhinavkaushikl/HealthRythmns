import joblib
from sklearn.ensemble import RandomForestRegressor

def check_model_in_pkl(model_path):
    """Check and print the model's type and attributes loaded from the .pkl file."""
    try:
        # Load the model
        model = joblib.load(model_path)
        
        # Check the type of the model
        print(f"Model type: {type(model)}")

        # Check if it's a RandomForestRegressor (or any other model you expect)
        if isinstance(model, RandomForestRegressor):
            print("It is a RandomForestRegressor model.")
        else:
            print("The loaded model is not a RandomForestRegressor.")
        
        # Optionally, you can inspect some attributes of the model
        print(f"Model attributes: {dir(model)}")
        
        # Check if the model has the 'predict' method
        if hasattr(model, 'predict'):
            print("The model has the 'predict' method.")
        else:
            print("The model does not have the 'predict' method.")
            
    except Exception as e:
        print(f"Error loading the model: {e}")

# Replace with your model's path
model_path = r'C:\Users\kau75421\OneDrive - Viavi Solutions Inc\Desktop\ML AI Reference Document\healthyfy\mobilepllc\models\final_randomforest_model.pkl'

check_model_in_pkl(model_path)