Work In Progress: Updating weekly
Data Insights Web App
This project is a Flask-based web application that allows users to upload CSV files, analyze the data, and perform various machine learning tasks such as Linear Regression,
Logistic Regression,
Decision Tree,
Random Forest,
Support Vector Machine (SVM),
K-Means Clustering,
Neural Network,
Gradient Boosting,
XGBoost,
K-Nearest Neighbors (KNN). The app provides an intuitive user interface for file uploads and delivers interactive insights based on the uploaded data.

Features
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
Interactive Visualizations:

Generates dynamic charts and graphs using Matplotlib and Seaborn.
Visualizes trends, predictions, and relationships within the data.
Responsive User Interface:

Designed with a modern UI, using HTML, CSS, and Flask templates.
Navigation options for seamless movement between tasks and pages.
Tech Stack
Backend: Flask, SQLAlchemy
Frontend: HTML, CSS, Jinja2 Templates
Database: PostgreSQL
Machine Learning: Scikit-learn, Statsmodels
Visualization: Matplotlib, Seaborn
Environment Management: Python virtual environments (venv)
How It Works
Upload a File:

Navigate to the Dashboard.
Drag and drop a CSV file or select one manually.
The app validates the file and stores metadata in PostgreSQL.


The app processes the dataset, applies relevant algorithms, and generates insights.
Visualizations and results are displayed dynamically on the corresponding task page.
Database Integration:

Each uploaded file is logged in the PostgreSQL database.
File metadata such as name, upload time, and task results are stored for future reference.
