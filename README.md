Data Insights Web App
This project is a Flask-based web application that allows users to upload CSV files, analyze the data, and perform various machine learning tasks such as Trend Analysis, Classification, Forecasting, Anomaly Detection, and Correlation Analysis. The app provides an intuitive user interface for file uploads and delivers interactive insights based on the uploaded data.

Features
CSV File Upload:

Users can upload CSV files via a drag-and-drop interface or by selecting files manually.
Validation ensures only CSV files are allowed.
Machine Learning Tasks:

Trend Analysis: Identifies trends over time using time-series data.
Classification: Classifies data using machine learning algorithms (e.g., Decision Trees).
Forecasting: Predicts future trends using statistical models.
Anomaly Detection: Identifies outliers in the dataset.
Correlation Analysis: Displays relationships between numerical variables using correlation matrices.
Data Preview:

Displays the first few rows of the uploaded dataset for user verification.
Provides insights into the structure and columns of the data.
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
Choose a Task:

Select from Trend Analysis, Classification, Forecasting, Anomaly Detection, or Correlation Analysis.
Perform Analysis:

The app processes the dataset, applies relevant algorithms, and generates insights.
Visualizations and results are displayed dynamically on the corresponding task page.
Database Integration:

Each uploaded file is logged in the PostgreSQL database.
File metadata such as name, upload time, and task results are stored for future reference.
