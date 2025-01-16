from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

def knn(X_train, X_test, y_train, y_test):
    model = KNeighborsClassifier() if y_train.nunique() > 2 else KNeighborsRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    if y_train.nunique() > 2:
        accuracy = accuracy_score(y_test, predictions)
        return {
            "status": "success",
            "message": f"K-Nearest Neighbors completed. Accuracy: {accuracy:.2f}"
        }
    else:
        mse = mean_squared_error(y_test, predictions)
        return {
            "status": "success",
            "message": f"K-Nearest Neighbors completed. Mean Squared Error: {mse:.2f}"
        }
