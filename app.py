from flask import Flask, render_template, request, redirect, url_for, flash
import os

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'  # Directory to save uploaded files
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16 MB limit
app.secret_key = 'your_secret_key'

# Ensure the upload folder exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

@app.route('/')
def index():
    return render_template('Dashboard.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        flash('No file part')
        return redirect(url_for('index'))

    file = request.files['file']
    if file.filename == '':
        flash('No selected file')
        return redirect(url_for('index'))

    if file:
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
        file.save(filepath)
        flash(f'File \"{file.filename}\" uploaded successfully!')
        # Redirect to predictions page
        return redirect(url_for('predictions'))

@app.route('/predictions')
def predictions():
    return render_template('predictions.html')

@app.route('/trend_analysis')
def trend_analysis():
    return render_template('trend_analysis.html')


if __name__ == "__main__":
    app.run(debug=True)
