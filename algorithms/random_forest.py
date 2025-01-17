from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def generate_classification_plot(model, X, y):
    # Reduce to 2D for plotting
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)
    x_min, x_max = X_reduced[:, 0].min() - 1, X_reduced[:, 0].max() + 1
    y_min, y_max = X_reduced[:, 1].min() - 1, X_reduced[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(pca.inverse_transform(np.c_[xx.ravel(), yy.ravel()]))
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8)
    plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=y, edgecolor='k')
    plt.title("Random Forest Classifier Decision Boundary")
    plt.xlabel("PCA Feature 1")
    plt.ylabel("PCA Feature 2")

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    return base64.b64encode(buffer.read()).decode('utf-8')

def generate_regression_plot(y_test, predictions):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, color='blue', label='Predictions vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Fit')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("Random Forest Regression Results")
    plt.legend()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    return base64.b64encode(buffer.read()).decode('utf-8')

def random_forest(X_train, X_test, y_train, y_test):
    is_classification = y_train.nunique() > 2
    model = RandomForestClassifier(random_state=42) if is_classification else RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    if is_classification:
        accuracy = accuracy_score(y_test, predictions)
        plot = generate_classification_plot(model, X_test, y_test)
        return {
            "status": "success",
            "message": f"Random Forest Classifier completed. Accuracy: {accuracy:.2f}",
            "plot": plot
        }
    else:
        mse = mean_squared_error(y_test, predictions)
        plot = generate_regression_plot(y_test, predictions)
        return {
            "status": "success",
            "message": f"Random Forest Regressor completed. Mean Squared Error: {mse:.2f}",
            "plot": plot
        }
