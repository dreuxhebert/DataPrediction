import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, confusion_matrix, mean_squared_error
from sklearn.neural_network import MLPClassifier, MLPRegressor
import importlib
import matplotlib.pyplot as plt
import seaborn as sns
import base64
from io import BytesIO

def encode_plot():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    return base64.b64encode(buffer.read()).decode('utf-8')

def generate_nn_classification_plot(y_test, predictions, class_names=None):
    cm = confusion_matrix(y_test, predictions)
    if class_names is None:
        class_names = sorted(set(y_test))

    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted Class')
    plt.ylabel('Actual Class')
    plt.title('Confusion Matrix')
    return encode_plot()

def generate_nn_regression_plot(y_test, predictions):
    plt.figure(figsize=(10, 6))
    plt.plot(range(len(y_test)), y_test, label='Actual', color='blue', linewidth=2)
    plt.plot(range(len(predictions)), predictions, label='Predicted', color='orange', linestyle='--', linewidth=2)
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.title('Neural Network Regression Results: Actual vs Predicted')
    plt.legend()
    return encode_plot()

def neural_network(X_train, X_test, y_train, y_test, task):
    if task == "classification":
        return neural_network_classification(X_train, X_test, y_train, y_test)
    elif task == "regression":
        return neural_network_regression(X_train, X_test, y_train, y_test)
    else:
        return {
            "status": "error",
            "message": "Invalid task specified. Use 'classification' or 'regression'."
        }

def neural_network_classification(X_train, X_test, y_train, y_test, class_names=None):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    try:
        model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)
        plot_base64 = generate_nn_classification_plot(y_test, predictions, class_names)

        return {
            "status": "success",
            "message": f"Neural Network Classification completed. Accuracy: {accuracy:.2f}",
            "plot": plot_base64
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error in Neural Network Classification: {str(e)}",
        }

def neural_network_regression(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    try:
        model = MLPRegressor(hidden_layer_sizes=(100,), max_iter=500, random_state=42)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        mse = mean_squared_error(y_test, predictions)
        plot_base64 = generate_nn_regression_plot(y_test, predictions)

        return {
            "status": "success",
            "message": f"Neural Network Regression completed. Mean Squared Error: {mse:.2f}",
            "plot": plot_base64
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error in Neural Network Regression: {str(e)}",
        }

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
        task = "classification" if y.nunique() <= 10 else "regression"

        module_name = f"algorithms.{algorithm}"
        try:
            module = importlib.import_module(module_name)
            algorithm_function = getattr(module, algorithm)
            if algorithm == "neural_network":
                return algorithm_function(X_train, X_test, y_train, y_test, task)
            else:
                return algorithm_function(X_train, X_test, y_train, y_test)
        except ModuleNotFoundError:
            return {"status": "error", "message": f"Algorithm '{algorithm}' not supported"}
        except AttributeError:
            return {"status": "error", "message": f"Algorithm function '{algorithm}' not found in module"}
    except Exception as e:
        return {"status": "error", "message": f"Error during {algorithm}: {str(e)}"}
