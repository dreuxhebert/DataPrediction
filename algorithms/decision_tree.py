from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor, plot_tree
from sklearn.metrics import accuracy_score, mean_squared_error
import matplotlib.pyplot as plt
import base64
from io import BytesIO

def encode_plot():
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()
    return base64.b64encode(buffer.read()).decode('utf-8')

def decision_tree(X_train, X_test, y_train, y_test):
    if y_train.nunique() > 2:

        model = DecisionTreeClassifier()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        accuracy = accuracy_score(y_test, predictions)


        plt.figure(figsize=(12, 8))
        plot_tree(model, feature_names=X_train.columns, class_names=[str(c) for c in model.classes_], filled=True)
        plt.title("Decision Tree Classifier")
        plot_base64 = encode_plot()

        return {
            "status": "success",
            "message": f"Decision Tree Classifier completed. Accuracy: {accuracy:.2f}",
            "plot": plot_base64
        }
    else:

        model = DecisionTreeRegressor()
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
        mse = mean_squared_error(y_test, predictions)


        plt.figure(figsize=(12, 8))
        plot_tree(model, feature_names=X_train.columns, filled=True)
        plt.title("Decision Tree Regressor")
        plot_base64 = encode_plot()

        return {
            "status": "success",
            "message": f"Decision Tree Regressor completed. Mean Squared Error: {mse:.2f}",
            "plot": plot_base64
        }
