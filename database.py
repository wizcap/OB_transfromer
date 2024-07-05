import sqlite3
import json
import logging
from datetime import datetime
from config import DATA_DIR, PREDICTION_RESULTS_FILE


class Database:
    def __init__(self):
        self.db_file = f"{DATA_DIR}/predictions.db"
        self.conn = None
        self.cursor = None

    def connect(self):
        try:
            self.conn = sqlite3.connect(self.db_file)
            self.cursor = self.conn.cursor()
            self.create_tables()
            logging.info("Successfully connected to the database.")
        except sqlite3.Error as e:
            logging.error(f"Error connecting to database: {e}")

    def close(self):
        if self.conn:
            self.conn.close()
            logging.info("Database connection closed.")

    def create_tables(self):
        try:
            self.cursor.execute('''
                CREATE TABLE IF NOT EXISTS predictions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    current_price REAL NOT NULL,
                    predicted_change REAL NOT NULL,
                    predicted_price REAL NOT NULL,
                    market_price REAL NOT NULL,
                    prediction_trend TEXT NOT NULL,
                    sma REAL NOT NULL
                )
            ''')
            self.conn.commit()
            logging.info("Predictions table created or already exists.")
        except sqlite3.Error as e:
            logging.error(f"Error creating table: {e}")

    def save_prediction(self, prediction_data):
        try:
            self.cursor.execute('''
                INSERT INTO predictions 
                (timestamp, current_price, predicted_change, predicted_price, market_price, prediction_trend, sma)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            ''', (
                prediction_data['timestamp'],
                prediction_data['current_price'],
                prediction_data['predicted_change'],
                prediction_data['predicted_price'],
                prediction_data['market_price'],
                prediction_data['prediction_trend'],
                prediction_data['sma']
            ))
            self.conn.commit()
            logging.info(f"Prediction saved: {prediction_data['timestamp']}")

            # Also save to JSON file for backward compatibility
            self.save_to_json(prediction_data)
        except sqlite3.Error as e:
            logging.error(f"Error saving prediction: {e}")

    def save_to_json(self, prediction_data):
        try:
            with open(PREDICTION_RESULTS_FILE, 'a') as f:
                json.dump(prediction_data, f)
                f.write('\n')
        except IOError as e:
            logging.error(f"Error saving prediction to JSON file: {e}")

    def get_recent_predictions(self, limit=100):
        try:
            self.cursor.execute('''
                SELECT * FROM predictions
                ORDER BY timestamp DESC
                LIMIT ?
            ''', (limit,))
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            logging.error(f"Error fetching recent predictions: {e}")
            return []

    def get_predictions_in_timeframe(self, start_time, end_time):
        try:
            self.cursor.execute('''
                SELECT * FROM predictions
                WHERE timestamp BETWEEN ? AND ?
                ORDER BY timestamp ASC
            ''', (start_time, end_time))
            return self.cursor.fetchall()
        except sqlite3.Error as e:
            logging.error(f"Error fetching predictions in timeframe: {e}")
            return []


# 创建一个全局的数据库实例
db = Database()


def init_db():
    db.connect()


def close_db():
    db.close()


# 在程序启动时初始化数据库连接
init_db()