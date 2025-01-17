import base64
import matplotlib.pyplot as plt
from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.decomposition import PCA
from io import BytesIO
import numpy as np
import pandas as pd

def gradient_boosting(X_train, X_test, y_train, y_test):
   
    if isinstance(X_train, pd.DataFrame):
        X_train = X_train.values
    if isinstance(X_test, pd.DataFrame):
        X_test = X_test.values
    if isinstance(y_train, pd.Series):
        y_train = y_train.values
    if isinstance(y_test, pd.Series):
        y_test = y_test.values

    model = GradientBoostingClassifier() if len(np.unique(y_train)) > 2 else GradientBoostingRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    if len(np.unique(y_train)) > 2:
        accuracy = accuracy_score(y_test, predictions)

        if X_train.shape[1] > 2:
            pca = PCA(n_components=2, random_state=42)
            X_train_2d = pca.fit_transform(X_train)
            X_test_2d = pca.transform(X_test)
        else:
            X_train_2d = X_train
            X_test_2d = X_test

        x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
        y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
        Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
        Z = Z.reshape(xx.shape)

        plt.contourf(xx, yy, Z, alpha=0.8)
        plt.scatter(X_test_2d[:, 0], X_test_2d[:, 1], c=y_test, edgecolor='k')
        plt.title("Gradient Boosting Decision Boundary (PCA Reduced)")
        plt.xlabel("PCA Feature 1")
        plt.ylabel("PCA Feature 2")

    else:
        mse = mean_squared_error(y_test, predictions)

        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, predictions, color='blue', label='Predictions vs Actual')
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Fit')
        plt.xlabel('Actual Values')
        plt.ylabel('Predicted Values')
        plt.title('Gradient Boosting Regression Results')
        plt.legend()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    plot_base64 = base64.b64encode(buffer.read()).decode('utf-8')

    return {
        "status": "success",
        "message": f"Gradient Boosting completed. {'Accuracy: {:.2f}'.format(accuracy) if len(np.unique(y_train)) > 2 else 'Mean Squared Error: {:.2f}'.format(mse)}",
        "plot": plot_base64
    }
