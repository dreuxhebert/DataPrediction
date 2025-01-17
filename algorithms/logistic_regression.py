import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
import base64
from io import BytesIO
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

def generate_plot(y_test, predictions):

    actual_counts = dict(zip(*np.unique(y_test, return_counts=True)))
    predicted_counts = dict(zip(*np.unique(predictions, return_counts=True)))

    classes = sorted(set(actual_counts.keys()).union(predicted_counts.keys()))

    actual_values = [actual_counts.get(cls, 0) for cls in classes]
    predicted_values = [predicted_counts.get(cls, 0) for cls in classes]

    x = np.arange(len(classes))

    plt.figure(figsize=(10, 6))
    plt.bar(x - 0.2, actual_values, width=0.4, label='Actual', color='blue', alpha=0.7)
    plt.bar(x + 0.2, predicted_values, width=0.4, label='Predicted', color='orange', alpha=0.7)

    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.title('Logistic Regression Results: Actual vs Predicted')
    plt.xticks(x, classes)
    plt.legend()

    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    return base64.b64encode(buffer.read()).decode('utf-8')

def logistic_regression(X_train, X_test, y_train, y_test):
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    try:

        model = LogisticRegression()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        accuracy = accuracy_score(y_test, predictions)

        plot_base64 = generate_plot(y_test, predictions)

        return {
            "status": "success",
            "message": f"Logistic Regression completed. Accuracy: {accuracy:.2f}",
            "plot": plot_base64
        }
    except Exception as e:
        return {
            "status": "error",
            "message": f"Error in Logistic Regression: {str(e)}",
        }
