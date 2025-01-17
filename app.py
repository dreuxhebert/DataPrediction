from flask import Flask, render_template, jsonify, request
from flask_sqlalchemy import SQLAlchemy
import os
from prediction_logic import run_predictions
from werkzeug.utils import secure_filename

app = Flask(__name__)
uploaded_file_path = None

@app.route('/run_predictions', methods=['POST'])
def handle_predictions():
    global uploaded_file_path  

    
    if not uploaded_file_path:
        return jsonify({"status": "error", "message": "No file uploaded"}), 400

    data = request.json
    algorithm = data.get('algorithm')

    if not algorithm:
        return jsonify({"status": "error", "message": "Algorithm not selected"}), 400

    
    result = run_predictions(uploaded_file_path, algorithm)
    return jsonify(result)


UPLOAD_FOLDER = 'uploads'
ALLOWED_EXTENSIONS = {'csv'}

app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS
# Route for file upload

@app.route('/upload_csv', methods=['POST'])
def upload_csv():
    global uploaded_file_path  # Declare global variable
    if 'file' not in request.files:
        return jsonify({"status": "error", "message": "No file part"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"status": "error", "message": "No selected file"}), 400

    if file and file.filename.endswith('.csv'):
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], secure_filename(file.filename))
        file.save(filepath)
        uploaded_file_path = filepath  # Update global variable
        return jsonify({"status": "success", "message": f"File {file.filename} uploaded successfully!", "filepath": filepath}), 200

    return jsonify({"status": "error", "message": "Invalid file type. Only CSV files are allowed."}), 400


app.config['SQLALCHEMY_DATABASE_URI'] = 'postgresql://USERNAME:YOURPASSWORD@localhost:PORT/MLearn'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


# API endpoint to handle ML algorithm selection
@app.route('/set_algorithm', methods=['POST'])
def set_algorithm():
    selected_algorithm = request.json.get('algorithm')
    if not selected_algorithm:
        return jsonify({"status": "error", "message": "No algorithm selected"}), 400

    print(f"Algorithm selected: {selected_algorithm}")
    return jsonify({"status": "success", "selected_algorithm": selected_algorithm})



class User(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)


@app.route('/database')
def database():
    try:
      
        user_count = User.query.count()
        return f"Connected to PostgreSQL. Total users: {user_count}"
    except Exception as e:
        return f"Database connection failed: {e}"


@app.route('/')
def home():
    return render_template('dashboard.html')


if __name__ == '__main__':
    app.run(debug=True)
