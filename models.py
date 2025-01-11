from flask_sqlalchemy import SQLAlchemy
from sqlalchemy import Column, Integer, String, DateTime

db = SQLAlchemy()  # Do not bind the app yet


class File(db.Model):
    __tablename__ = 'file'  # Table name in PostgreSQL
    id = db.Column(db.Integer, primary_key=True)
    filename = db.Column(db.String(120), nullable=False)
    uploaded_at = db.Column(db.DateTime, default=datetime.utcnow)
    trend_analysis_result = db.Column(db.Text, nullable=True)
    classification_result = db.Column(db.Text, nullable=True)
    forecasting_result = db.Column(db.Text, nullable=True)
    anomaly_detection_result = db.Column(db.Text, nullable=True)
    correlation_analysis_result = db.Column(db.Text, nullable=True)

    def __repr__(self):
        return f"<File {self.filename}>"
