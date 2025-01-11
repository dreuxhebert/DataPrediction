import os
import pandas as pd
from flask import Flask, render_template, request, redirect, url_for, flash, session
from flask_sqlalchemy import SQLAlchemy
from models import File
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from statsmodels.tsa.holtwinters import SimpleExpSmoothing
from sklearn.ensemble import IsolationForest
import seaborn as sns
import io
import base64
import matplotlib.pyplot as plt

app = Flask(__name__)
app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql+pg8000://postgres:123@localhost:5432/Datadata'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'  # Directory to save uploaded files
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
app.secret_key = 'your_secret_key'

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

db = SQLAlchemy(app)

with app.app_context():
    db.create_all()


def allowed_file(filename):
    """Check if a file is a CSV."""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'csv'


@app.route('/')
def index():
    return render_template('Dashboard.html')


@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part', 'error')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file', 'error')
        return redirect(url_for('index'))

    if file and allowed_file(file.filename):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)

        try:

            data = pd.read_csv(filepath)

            session['data'] = data.to_json()

            new_file = File(filename=file.filename, filepath=filepath)
            db.session.add(new_file)
            db.session.commit()

            flash(f'File "{file.filename}" uploaded successfully!', 'success')
            return redirect(url_for('predictions'))
        except Exception as e:
            print(f"Error reading CSV file: {e}")
            flash('Error reading the CSV file. Please ensure it is in the correct format.', 'error')
            return redirect(url_for('index'))
    else:
        flash('Invalid file type! Please upload a CSV file.', 'error')
        return redirect(url_for('index'))


@app.route('/predictions')
def predictions():
    return render_template('predictions.html')


@app.route('/trend_analysis')
def trend_analysis():
    if 'data' not in session:
        flash('No data available. Please upload a CSV file first.', 'error')
        return redirect(url_for('index'))

    data = pd.read_json(session['data'])

    time_column = None
    for col in data.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            time_column = col
            break

    if time_column:
        data[time_column] = pd.to_datetime(data[time_column], errors='coerce')
        data = data.dropna(subset=[time_column])
        data = data.sort_values(by=time_column)
    else:
        flash('No time-related column found. Trend analysis may not be meaningful.', 'warning')

    trend_plots = []
    for col in data.select_dtypes(include=['number']).columns:
        if time_column:

            plt.figure(figsize=(10, 5))
            plt.plot(data[time_column], data[col], label=col, marker='o')
            plt.title(f"Trend Analysis: {col}")
            plt.xlabel(time_column)
            plt.ylabel(col)
            plt.legend()
            plt.tight_layout()

            buf = io.BytesIO()
            plt.savefig(buf, format='png')
            buf.seek(0)
            plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
            buf.close()
            trend_plots.append(plot_data)
            plt.close()
        else:
            flash('Time-related column missing; cannot perform time-series analysis.', 'error')

    return render_template('trend_analysis.html', columns=list(data.columns), trend_plots=trend_plots)


@app.route('/classification')
def classification():
    if 'data' not in session:
        flash('No data available. Please upload a CSV file first.', 'error')
        return redirect(url_for('index'))

    data = pd.read_json(session['data'])

    target_column = data.columns[-1]
    X = data.iloc[:, :-1]
    y = data[target_column]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    clf = DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    accuracy = clf.score(X_test, y_test)

    return render_template(
        'classification.html',
        columns=list(data.columns),
        preview=data.head().to_html(),
        target=target_column,
        accuracy=accuracy
    )


@app.route('/forecasting')
def forecasting():
    if 'data' not in session:
        flash('No data available. Please upload a CSV file first.', 'error')
        return redirect(url_for('index'))

    data = pd.read_json(session['data'])

    time_column = None
    for col in data.columns:
        if 'date' in col.lower() or 'time' in col.lower():
            time_column = col
            break

    if time_column:
        data[time_column] = pd.to_datetime(data[time_column], errors='coerce')
        data = data.dropna(subset=[time_column])
        data = data.sort_values(by=time_column)

        forecast_column = data.select_dtypes(include=['number']).columns[0]
        model = SimpleExpSmoothing(data[forecast_column]).fit()
        forecast = model.forecast(5)

        return render_template(
            'forecasting.html',
            columns=list(data.columns),
            preview=data.head().to_html(),
            forecast=forecast.to_list(),
            time_column=time_column
        )
    else:
        flash('No time-related column found for forecasting.', 'error')
        return redirect(url_for('index'))


@app.route('/anomaly_detection')
def anomaly_detection():
    if 'data' not in session:
        flash('No data available. Please upload a CSV file first.', 'error')
        return redirect(url_for('index'))

    data = pd.read_json(session['data'])

    model = IsolationForest(random_state=42)
    numerical_data = data.select_dtypes(include=['number'])
    model.fit(numerical_data)
    anomalies = model.predict(numerical_data)
    anomaly_indices = (anomalies == -1)

    data['Anomaly'] = anomaly_indices

    return render_template(
        'anomaly_detection.html',
        columns=list(data.columns),
        preview=data.head().to_html(),
        anomalies=data[data['Anomaly']].to_html()
    )


@app.route('/correlation_analysis')
def correlation_analysis():
    if 'data' not in session:
        flash('No data available. Please upload a CSV file first.', 'error')
        return redirect(url_for('index'))

    data = pd.read_json(session['data'])

    # x
    correlation_matrix = data.corr()

    plt.figure(figsize=(10, 8))
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Correlation Analysis")

    buf = io.BytesIO()
    plt.savefig(buf, format='png')
    buf.seek(0)
    plot_data = base64.b64encode(buf.getvalue()).decode('utf-8')
    buf.close()
    plt.close()

    return render_template(
        'correlation_analysis.html',
        columns=list(data.columns),
        preview=data.head().to_html(),
        heatmap=plot_data
    )


if __name__ == "__main__":
    with app.app_context():
        db.create_all()  # Create tables in the database
        print("Database tables created.")
    app.run(debug=True)
