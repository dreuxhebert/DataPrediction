import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import importlib

def run_predictions(filepath, algorithm):
    
    try:
        df = pd.read_csv(filepath)
    except Exception as e:
        return {"status": "error", "message": f"Failed to load CSV: {str(e)}"}

   
    if df.shape[1] < 2:
        return {"status": "error", "message": "CSV file must contain at least two columns"}

   
    try:
        
        for column in df.select_dtypes(include=['object']).columns:
            le = LabelEncoder()
            df[column] = le.fit_transform(df[column])

        X = df.iloc[:, :-1]
        y = df.iloc[:, -1]
    except Exception as e:
        return {"status": "error", "message": f"Error during preprocessing: {str(e)}"}

    try:
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Dynamically import and execute the selected algorithm
        module_name = f"algorithms.{algorithm}"
        try:
            module = importlib.import_module(module_name)
            algorithm_function = getattr(module, algorithm)
            return algorithm_function(X_train, X_test, y_train, y_test)
        except ModuleNotFoundError:
            return {"status": "error", "message": f"Algorithm '{algorithm}' not supported"}
        except AttributeError:
            return {"status": "error", "message": f"Algorithm function '{algorithm}' not found in module"}

    except Exception as e:
        return {"status": "error", "message": f"Error during {algorithm}: {str(e)}"}
