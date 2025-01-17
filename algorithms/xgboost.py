from xgboost import XGBClassifier, XGBRegressor
from sklearn.metrics import accuracy_score, mean_squared_error
from sklearn.preprocessing import LabelEncoder
from sklearn.decomposition import PCA
import numpy as np
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def generate_plot(x_min, x_max, y_min, y_max, xx, yy, Z, X, y, title):
    plt.contourf(xx, yy, Z, alpha=0.8, cmap="viridis")
    plt.scatter(X[:, 0], X[:, 1], c=y, edgecolor="k", cmap="viridis")
    plt.title(title)
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")

    buffer = BytesIO()
    plt.savefig(buffer, format="png")
    buffer.seek(0)
    plt.close()
    return base64.b64encode(buffer.read()).decode("utf-8")


def xgboost(X_train, X_test, y_train, y_test):
    label_encoder = LabelEncoder()

    if len(X_train.shape) > 2 or len(X_test.shape) > 2:
        return {"status": "error", "message": "Data format is invalid. Ensure X_train and X_test have 2 dimensions."}

    if X_train.shape[1] > 2:
        pca = PCA(n_components=2)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

    if y_train.nunique() > 2:
        y_train_encoded = label_encoder.fit_transform(y_train)
        y_test_encoded = label_encoder.transform(y_test)

        model = XGBClassifier(eval_metric='logloss')
        model.fit(X_train, y_train_encoded)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test_encoded, predictions)

        x_min, x_max = X_train[:, 0].min() - 1, X_train[:, 0].max() + 1
        y_min, y_max = X_train[:, 1].min() - 1, X_train[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
        Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
        Z = Z.reshape(xx.shape)

        plot_base64 = generate_plot(
            x_min, x_max, y_min, y_max, xx, yy, Z, X_test, y_test_encoded, "XGBoost Classifier Decision Boundary"
        )

        return {
            "status": "success",
            "message": f"XGBoost completed. Accuracy: {accuracy:.2f}",
            "plot": plot_base64
        }

    else:
        model = XGBRegressor()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)

        plt.figure(figsize=(8, 6))
        plt.scatter(y_test, predictions, color="blue", label="Predictions vs Actual")
        plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color="red", linestyle="--", label="Ideal Fit")
        plt.xlabel("Actual Values")
        plt.ylabel("Predicted Values")
        plt.title("XGBoost Regressor Results")
        plt.legend()

        buffer = BytesIO()
        plt.savefig(buffer, format="png")
        buffer.seek(0)
        plt.close()
        plot_base64 = base64.b64encode(buffer.read()).decode("utf-8")

        return {
            "status": "success",
            "message": f"XGBoost completed. Mean Squared Error: {mse:.2f}",
            "plot": plot_base64
        }
