"""SQLite database management for EEG system."""
import sqlite3
import json
from datetime import datetime
import os

class EEGDatabase:
    def __init__(self, db_path="eeg_system.db"):
        self.db_path = db_path
        self.init_database()
        
    def init_database(self):
        """Initialize database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Users table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS users (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT UNIQUE NOT NULL,
                subject_id INTEGER NOT NULL,
                data_segments INTEGER,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # Authentication logs
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS auth_logs (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                username TEXT NOT NULL,
                success BOOLEAN NOT NULL,
                confidence REAL,
                reason TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        # System settings
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS settings (
                key TEXT PRIMARY KEY,
                value TEXT NOT NULL,
                updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        ''')
        
        conn.commit()
        conn.close()
        
    def add_user(self, username, subject_id, data_segments=0):
        """Add new user to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        try:
            cursor.execute('''
                INSERT INTO users (username, subject_id, data_segments)
                VALUES (?, ?, ?)
            ''', (username, subject_id, data_segments))
            conn.commit()
            return True
        except sqlite3.IntegrityError:
            return False
        finally:
            conn.close()
            
    def remove_user(self, username):
        """Remove user from database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('DELETE FROM users WHERE username = ?', (username,))
        affected = cursor.rowcount
        conn.commit()
        conn.close()
        
        return affected > 0
        
    def get_users(self):
        """Get all users."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('SELECT username, subject_id, data_segments FROM users')
        users = cursor.fetchall()
        conn.close()
        
        return {user[0]: {'subject_id': user[1], 'data_segments': user[2]} for user in users}
        
    def log_authentication(self, username, success, confidence=None, reason=None):
        """Log authentication attempt."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            INSERT INTO auth_logs (username, success, confidence, reason)
            VALUES (?, ?, ?, ?)
        ''', (username, success, confidence, reason))
        
        conn.commit()
        conn.close()
        
    def get_auth_stats(self):
        """Get authentication statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute('''
            SELECT 
                COUNT(*) as total_attempts,
                SUM(CASE WHEN success = 1 THEN 1 ELSE 0 END) as successful,
                AVG(confidence) as avg_confidence
            FROM auth_logs
        ''')
        
        stats = cursor.fetchone()
        conn.close()
        
        return {
            'total_attempts': stats[0],
            'successful': stats[1],
            'success_rate': stats[1] / stats[0] if stats[0] > 0 else 0,
            'avg_confidence': stats[2] or 0
        }

# Global database instance
db = EEGDatabase()