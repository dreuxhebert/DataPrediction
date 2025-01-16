from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

def random_forest(X_train, X_test, y_train, y_test):
    model = RandomForestClassifier(random_state=42) if y_train.nunique() > 2 else RandomForestRegressor(random_state=42)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    if y_train.nunique() > 2:
        accuracy = accuracy_score(y_test, predictions)
        return {
            "status": "success",
            "message": f"Random Forest Classifier completed. Accuracy: {accuracy:.2f}"
        }
    else:
        mse = mean_squared_error(y_test, predictions)
        return {
            "status": "success",
            "message": f"Random Forest Regressor completed. Mean Squared Error: {mse:.2f}"
        }
