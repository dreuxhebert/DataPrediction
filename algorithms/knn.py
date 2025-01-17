from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64
import pandas as pd

def knn(X_train, X_test, y_train, y_test):

    if isinstance(X_train, np.ndarray):
        X_train = pd.DataFrame(X_train, columns=[f"Feature {i+1}" for i in range(X_train.shape[1])])
    if isinstance(X_test, np.ndarray):
        X_test = pd.DataFrame(X_test, columns=X_train.columns)


    X_train = X_train.iloc[:, :2]
    X_test = X_test.iloc[:, :2]


    is_classification = y_train.nunique() > 2
    model = KNeighborsClassifier() if is_classification else KNeighborsRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)


    plt.figure(figsize=(8, 6))
    if is_classification:

        x_min, x_max = X_train.iloc[:, 0].min() - 1, X_train.iloc[:, 0].max() + 1
        y_min, y_max = X_train.iloc[:, 1].min() - 1, X_train.iloc[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                             np.arange(y_min, y_max, 0.01))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)
        plt.contourf(xx, yy, Z, alpha=0.8, cmap='coolwarm')
        plt.scatter(X_train.iloc[:, 0], X_train.iloc[:, 1], c=y_train, edgecolor='k', cmap='coolwarm')
        plt.title("K-Nearest Neighbors Classification")
        plt.xlabel("Feature 1")
        plt.ylabel("Feature 2")
    else:

        plt.scatter(y_test, predictions, color='blue', label='Predictions vs Actual')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Fit')
        plt.title("K-Nearest Neighbors Regression")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.legend()


    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()


    plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')


    if is_classification:
        accuracy = accuracy_score(y_test, predictions)
        return {
            "status": "success",
            "message": f"K-Nearest Neighbors completed. Accuracy: {accuracy:.2f}",
            "plot": plot_base64
        }
    else:
        mse = mean_squared_error(y_test, predictions)
        return {
            "status": "success",
            "message": f"K-Nearest Neighbors completed. Mean Squared Error: {mse:.2f}",
            "plot": plot_base64
        }
