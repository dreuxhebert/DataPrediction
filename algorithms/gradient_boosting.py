from sklearn.ensemble import GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.metrics import accuracy_score, mean_squared_error

def gradient_boosting(X_train, X_test, y_train, y_test):
    model = GradientBoostingClassifier() if y_train.nunique() > 2 else GradientBoostingRegressor()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    if y_train.nunique() > 2:
        accuracy = accuracy_score(y_test, predictions)
        return {
            "status": "success",
            "message": f"Gradient Boosting completed. Accuracy: {accuracy:.2f}"
        }
    else:
        mse = mean_squared_error(y_test, predictions)
        return {
            "status": "success",
            "message": f"Gradient Boosting completed. Mean Squared Error: {mse:.2f}"
        }
