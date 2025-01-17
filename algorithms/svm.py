from sklearn.svm import SVC, SVR
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import base64

def generate_svm_classification_plot(model, X, y):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))

    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.contourf(xx, yy, Z, alpha=0.8, cmap="coolwarm")
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor='k', cmap="coolwarm", s=20)
    plt.title("SVM Classifier Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    return base64.b64encode(buffer.read()).decode('utf-8')

def generate_svm_regression_plot(y_test, predictions):
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, color='blue', label='Predictions vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Fit')
    plt.xlabel("Actual Values")
    plt.ylabel("Predicted Values")
    plt.title("SVM Regression Results")
    plt.legend()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    return base64.b64encode(buffer.read()).decode('utf-8')

def svm(X_train, X_test, y_train, y_test):
    # Check PCA
    pca = None
    if X_train.shape[1] > 2:
        pca = PCA(n_components=2)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)


    is_classification = len(np.unique(y_train)) > 2
    model = SVC(kernel='rbf', probability=True, random_state=42) if is_classification else SVR(kernel='rbf')


    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    if is_classification:
        accuracy = accuracy_score(y_test, predictions)
        if pca:
            plot = generate_svm_classification_plot(model, X_test, y_test)
        else:
            plot = generate_svm_classification_plot(model, X_test, y_test)
        return {
            "status": "success",
            "message": f"SVM Classifier completed. Accuracy: {accuracy:.2f}",
            "plot": plot
        }
    else:
        mse = mean_squared_error(y_test, predictions)
        plot = generate_svm_regression_plot(y_test, predictions)
        return {
            "status": "success",
            "message": f"SVM Regressor completed. Mean Squared Error: {mse:.2f}",
            "plot": plot
        }
