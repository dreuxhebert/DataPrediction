Work In Progress: Updating weekly


DataPredict Web App

This project is a Flask-based web application that allows users to upload CSV files, analyze the data, and perform various machine learning tasks.
The app provides an intuitive user interface for file uploads and delivers interactive insights based on the uploaded data.


CSV File Upload:
Users can upload CSV files by selecting files manually.
Validation ensures only CSV files are allowed.

Machine Learning Tasks:

Linear Regression: Predicts a continuous target variable using one or more input features.
Logistic Regression: Predicts a binary or multiclass target variable based on input features.
Decision Tree: Constructs a tree structure to make predictions based on feature splits.
Random Forest: Uses multiple decision trees to improve prediction accuracy and robustness.
Support Vector Machine (SVM): Finds the optimal hyperplane for classifying data in higher-dimensional space.
K-Means Clustering: Groups data points into clusters based on similarity or proximity.
Neural Network: Models complex relationships in data using interconnected layers of neurons.
Gradient Boosting: Sequentially improves model predictions by minimizing error in a boosted manner.
XGBoost: Optimized implementation of gradient boosting for high performance and efficiency.
K-Nearest Neighbors (KNN): Predicts the target variable based on the closest data points in the feature space.



Data Preview:

Graphical Representation:
Displays visual insights into the dataset using graphs.
Bar charts for categorical data distribution.
Histograms for numerical data distribution.
Scatter plots for relationships between two selected variables.
Heatmaps to visualize correlations between numerical columns.


PostgreSQL Integration:

Stores metadata about uploaded files (e.g., filename, timestamp).
The PostgreSQL database is used to securely manage uploaded data.



Tech Stack:

Backend Framework: Flask 

ORM: SQLAlchemy

Frontend Languages: HTML, CSS, JavaScript

Templating Engine: Jinja2

Database: PostgreSQL
Tools: psycopg2 (PostgreSQL adapter for Python)

Machine Learning Libraries: 

Scikit-learn, XGBoost, Statsmodels
Clustering: K-Means (from Scikit-learn)

Classification/Regression: Decision Trees, Random Forests, SVM, Gradient Boosting, etc.

Visualization Plotting Libraries: 
Matplotlib, Seaborn

Virtual Environments: Python venv

Package Management: pip

Environment Management: Python virtual environments (venv)




