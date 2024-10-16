import sqlite3
import json
from datetime import datetime

def create_connection():
    conn = sqlite3.connect('monte_carlo_analyses.db')
    return conn

def create_table():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS analyses (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT NOT NULL,
            date TEXT NOT NULL,
            config TEXT NOT NULL,
            results TEXT NOT NULL
        )
    ''')
    conn.commit()
    conn.close()

def save_analysis(name, config, results):
    conn = create_connection()
    cursor = conn.cursor()
    date = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    cursor.execute('''
        INSERT INTO analyses (name, date, config, results)
        VALUES (?, ?, ?, ?)
    ''', (name, date, json.dumps(config), json.dumps(results)))
    conn.commit()
    conn.close()

def get_all_analyses():
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT id, name, date FROM analyses')
    analyses = cursor.fetchall()
    conn.close()
    return analyses

def get_analysis(analysis_id):
    conn = create_connection()
    cursor = conn.cursor()
    cursor.execute('SELECT * FROM analyses WHERE id = ?', (analysis_id,))
    analysis = cursor.fetchone()
    conn.close()
    if analysis:
        return {
            'id': analysis[0],
            'name': analysis[1],
            'date': analysis[2],
            'config': json.loads(analysis[3]),
            'results': json.loads(analysis[4])
        }
    return None

# Initialize the database
create_table()
