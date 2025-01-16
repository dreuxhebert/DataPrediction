from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

def decision_tree(X_train, X_test, y_train, y_test):
    model = DecisionTreeClassifier() if y_train.nunique() > 2 else DecisionTreeRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    if y_train.nunique() > 2:
        accuracy = accuracy_score(y_test, predictions)
        return {
            "status": "success",
            "message": f"Decision Tree Classifier completed. Accuracy: {accuracy:.2f}"
        }
    else:
        mse = mean_squared_error(y_test, predictions)
        return {
            "status": "success",
            "message": f"Decision Tree Regressor completed. Mean Squared Error: {mse:.2f}"
        }
