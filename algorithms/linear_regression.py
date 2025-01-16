from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import base64
from io import BytesIO


def generate_plot(y_test, predictions):
    
    plt.figure(figsize=(8, 6))
    plt.scatter(y_test, predictions, color='blue', label='Predictions vs Actual')
    plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', label='Ideal Fit')
    plt.xlabel('Actual Values')
    plt.ylabel('Predicted Values')
    plt.title('Linear Regression Results')
    plt.legend()

   
    buffer = BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    plt.close()

    return base64.b64encode(buffer.read()).decode('utf-8')


def linear_regression(X_train, X_test, y_train, y_test):
    model = LinearRegression()
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)

   
    plot_base64 = generate_plot(y_test, predictions)

    return {
        "status": "success",
        "message": f"Linear Regression completed. Mean Squared Error: {mse:.2f}",
        "plot": plot_base64  # Include the plot in the response
    }
