import os
import sys
import cv2
import numpy as np
import streamlit as st
import tensorflow as tf
import sqlite3
import json
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import gc
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.model_selection import train_test_split
from sklearn.metrics import (accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc)
from sklearn.svm import SVC
import warnings
from cryptography.fernet import Fernet
from cryptography.hazmat.primitives import hashes
from cryptography.hazmat.primitives.kdf.pbkdf2 import PBKDF2HMAC
from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes
from cryptography.hazmat.backends import default_backend
from base64 import b64encode, b64decode
import dlib
from scipy.spatial import distance
from collections import deque
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout


# Configure environment
os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'
warnings.filterwarnings('ignore')
tf.get_logger().setLevel('ERROR')

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('face_recognition.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ConfigManager:
    """Manages application configuration"""
    DEFAULT_CONFIG = {
        'min_face_size': (30, 30),
        'detection_confidence': 0.5,
        'feature_extraction': {
            'image_size': (224, 224),
            'batch_size': 32
        },
        'training': {
            'test_size': 0.2,
            'random_state': 42
        },
        'camera': {
            'capture_interval': 0.1,
            'frame_size': (640, 480)
        }
    }

    def __init__(self, config_path: Path = Path('config.json')):
        self.config_path = config_path
        self.config = self.load_config()

    def load_config(self) -> dict:
        """Load configuration from file or create default"""
        try:
            if self.config_path.exists():
                with open(self.config_path, 'r') as f:
                    return {**self.DEFAULT_CONFIG, **json.load(f)}
            return self.DEFAULT_CONFIG
        except Exception as e:
            logger.error(f"Error loading config: {e}")
            return self.DEFAULT_CONFIG

    def save_config(self) -> None:
        """Save current configuration to file"""
        try:
            with open(self.config_path, 'w') as f:
                json.dump(self.config, f, indent=4)
        except Exception as e:
            logger.error(f"Error saving config: {e}")

class EncryptionManager:
    def __init__(self, key_path: Path = Path('encryption_key.key')):
        """Initialize encryption manager with key handling
        Args:
            key_path (Path): Path to store/retrieve encryption key
        """
        self.key_path = key_path
        self.key = self._load_or_generate_key()
        self.fernet = Fernet(self.key)
        self.backend = default_backend()
        
    def _load_or_generate_key(self) -> bytes:
        """Load existing key or generate new one"""
        try:
            if self.key_path.exists():
                with open(self.key_path, 'rb') as key_file:
                    return key_file.read()
            else:
                key = Fernet.generate_key()
                with open(self.key_path, 'wb') as key_file:
                    key_file.write(key)
                return key
        except Exception as e:
            logging.error(f"Error handling encryption key: {e}")
            raise

    def encrypt_image(self, image: np.ndarray) -> bytes:
        """Encrypt image data
        Args:
            image (np.ndarray): Image array to encrypt
        Returns:
            bytes: Encrypted image data
        """
        try:
            # Convert image to bytes
            _, img_buffer = cv2.imencode('.jpg', image)
            img_bytes = img_buffer.tobytes()
            
            # Generate random IV
            iv = os.urandom(16)
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(self.key[:32]),  # Use first 32 bytes as AES key
                modes.CBC(iv),
                backend=self.backend
            )
            encryptor = cipher.encryptor()
            
            # Pad data to be multiple of 16 bytes
            pad_length = 16 - (len(img_bytes) % 16)
            padded_data = img_bytes + (bytes([pad_length]) * pad_length)
            
            # Encrypt data
            encrypted_data = encryptor.update(padded_data) + encryptor.finalize()
            
            # Combine IV and encrypted data
            return iv + encrypted_data
            
        except Exception as e:
            logging.error(f"Error encrypting image: {e}")
            raise

    def decrypt_image(self, encrypted_data: bytes) -> np.ndarray:
        """Decrypt image data
        Args:
            encrypted_data (bytes): Encrypted image data
        Returns:
            np.ndarray: Decrypted image array
        """
        try:
            # Extract IV and encrypted data
            iv = encrypted_data[:16]
            encrypted_image = encrypted_data[16:]
            
            # Create cipher
            cipher = Cipher(
                algorithms.AES(self.key[:32]),
                modes.CBC(iv),
                backend=self.backend
            )
            decryptor = cipher.decryptor()
            
            # Decrypt data
            padded_data = decryptor.update(encrypted_image) + decryptor.finalize()
            
            # Remove padding
            pad_length = padded_data[-1]
            img_bytes = padded_data[:-pad_length]
            
            # Convert back to image
            img_array = np.frombuffer(img_bytes, dtype=np.uint8)
            return cv2.imdecode(img_array, cv2.IMREAD_COLOR)  
        except Exception as e:
            logging.error(f"Error decrypting image: {e}")
            raise

    def secure_save_image(self, image: np.ndarray, file_path: Path) -> None:
        """Save encrypted image to file
        Args:
            image (np.ndarray): Image to save
            file_path (Path): Path to save encrypted image
        """
        try:
            encrypted_data = self.encrypt_image(image)
            with open(file_path, 'wb') as f:
                f.write(encrypted_data)
            logging.info(f"Securely saved image to {file_path}")
        except Exception as e:
            logging.error(f"Error saving encrypted image: {e}")
            raise

    def secure_load_image(self, file_path: Path) -> np.ndarray:
        """Load and decrypt image from file        
        Args:
            file_path (Path): Path to encrypted image
        Returns:
            np.ndarray: Decrypted image array
        """
        try:
            with open(file_path, 'rb') as f:
                encrypted_data = f.read()
            return self.decrypt_image(encrypted_data)
        except Exception as e:
            logging.error(f"Error loading encrypted image: {e}")
            raise

class DatabaseManager:
    def __init__(self, db_path: Path):
        """Initialize Database Manager 
        Args:
            db_path (Path): Path to the SQLite database file
        """
        self.db_path = db_path
        self._initialize_database()
        
    def _initialize_database(self) -> None:
        """Initialize SQLite database with required tables"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()

                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS authorized_users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        password TEXT NOT NULL,
                        role TEXT DEFAULT 'user',
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        last_login TIMESTAMP,
                        status TEXT DEFAULT 'active'
                    )
                ''')
                                
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        image_count INTEGER DEFAULT 0,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        status TEXT DEFAULT 'active'
                    )
                ''')
                

                # Create training sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS training_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        num_users INTEGER,
                        total_images INTEGER,
                        accuracy REAL,
                        precision_score REAL,
                        recall_score REAL,
                        f1_score REAL,
                        confusion_matrix TEXT,
                        model_path TEXT,
                        training_duration REAL
                    )
                ''')
                
                # Create recognition sessions table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS recognition_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_name TEXT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        user_id INTEGER,
                        confidence REAL,
                        duration REAL,
                        status TEXT DEFAULT 'active',
                        total_recognitions INTEGER DEFAULT 0,
                        unique_faces INTEGER DEFAULT 0,
                        avg_confidence REAL DEFAULT 0,
                        FOREIGN KEY(user_id) REFERENCES users(id)
                    )
                ''')
                
                # Create recognition logs table
                cursor.execute('''
                    CREATE TABLE IF NOT EXISTS recognition_logs (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id INTEGER,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        user_id INTEGER,
                        confidence REAL,
                        frame_path TEXT,
                        FOREIGN KEY(session_id) REFERENCES recognition_sessions(id),
                        FOREIGN KEY(user_id) REFERENCES users(id)
                    )
                ''')
                
                # Create view for session summaries
                cursor.execute('''
                    CREATE VIEW IF NOT EXISTS session_summaries AS
                    SELECT 
                        s.id,
                        s.session_name,
                        s.timestamp,
                        COUNT(DISTINCT l.user_id) as unique_faces,
                        COUNT(*) as total_recognitions,
                        AVG(l.confidence) as avg_confidence,
                        GROUP_CONCAT(DISTINCT u.name) as recognized_people
                    FROM recognition_sessions s
                    LEFT JOIN recognition_logs l ON s.id = l.session_id
                    LEFT JOIN users u ON l.user_id = u.id
                    GROUP BY s.id
                ''')
                
                # Insert default admin user
                cursor.execute("""
                    INSERT OR IGNORE INTO authorized_users (name, password, role)
                    VALUES ('admin', 'admin123', 'admin')
                """)
                
                conn.commit()
                logging.info("Database initialized successfully")
        
        except sqlite3.Error as e:
            logging.error(f"Database initialization error: {e}")
            raise

    def log_recognition(self, session_id: int, user_id: int, confidence: float, frame_path: Optional[str] = None) -> int:
        """Log recognition event
        
        Args:
            session_id (int): Recognition session ID
            user_id (int): Recognized user ID
            confidence (float): Recognition confidence score
            frame_path (Optional[str]): Path to saved frame image
            
        Returns:
            int: Recognition log ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO recognition_logs 
                    (session_id, user_id, confidence, frame_path)
                    VALUES (?, ?, ?, ?)
                """, (session_id, user_id, confidence, frame_path))
                log_id = cursor.lastrowid
                conn.commit()
                logging.info(f"Logged recognition event: {log_id}")
                return log_id
        except sqlite3.Error as e:
            logging.error(f"Error logging recognition: {e}")
            raise

        def close_recognition_session(self, session_id: int) -> None:
            """Close recognition session
                    Args:
                    session_id (int): Recognition session ID
            """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE recognition_sessions
                    SET status = 'completed',
                        duration = (
                            strftime('%s', 'now') - 
                            strftime('%s', timestamp)
                        )
                    WHERE id = ?
                """, (session_id,))
                conn.commit()
                logging.info(f"Closed recognition session: {session_id}")
        except sqlite3.Error as e:
            logging.error(f"Error closing recognition session: {e}")
            raise

    
    def authenticate_user(self, username: str, password: str) -> Optional[Dict[str, Any]]:
        """Authenticate user and return user details if successful"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, role, created_at, last_login
                    FROM authorized_users
                    WHERE name = ? AND password = ? AND status = 'active'
                """, (username, password))
                user = cursor.fetchone()
                
                if user:
                    # Update last login
                    cursor.execute("""
                        UPDATE authorized_users 
                        SET last_login = CURRENT_TIMESTAMP
                        WHERE id = ?
                    """, (user['id'],))
                    conn.commit()
                    return dict(user)
            return None
        except sqlite3.Error as e:
            logging.error(f"Authentication error: {e}")
            raise

    def get_recognition_stats_by_user(self, hours: int = 24) -> pd.DataFrame:
        """Get detailed recognition statistics grouped by user"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                query = """
                    SELECT 
                        u.name,
                        COUNT(*) as recognition_count,
                        AVG(r.confidence) as avg_confidence,
                        MIN(r.confidence) as min_confidence,
                        MAX(r.confidence) as max_confidence,
                        MIN(r.timestamp) as first_recognition,
                        MAX(r.timestamp) as last_recognition
                    FROM recognition_logs r
                    JOIN users u ON r.user_id = u.id
                    WHERE r.timestamp >= datetime('now', ?)
                    GROUP BY u.name
                    ORDER BY recognition_count DESC
                """
                return pd.read_sql_query(query, conn, params=(f'-{hours} hours',))
        except sqlite3.Error as e:
            logging.error(f"Error getting recognition stats: {e}")
            raise

    def add_user(self, name: str) -> int:
        """Add new user to database
        
        Args:
            name (str): Username
            
        Returns:
            int: User ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT OR IGNORE INTO users (name, last_updated) 
                    VALUES (?, CURRENT_TIMESTAMP)
                """, (name,))
                cursor.execute("SELECT id FROM users WHERE name = ?", (name,))
                user_id = cursor.fetchone()[0]
                conn.commit()
                logging.info(f"Added user: {name} with ID: {user_id}")
                return user_id
        except sqlite3.Error as e:
            logging.error(f"Error adding user: {e}")
            raise

    def get_user(self, user_id: int) -> Optional[Dict[str, Any]]:
        """Get user information
        
        Args:
            user_id (int): User ID
            
        Returns:
            Optional[Dict[str, Any]]: User information or None if not found
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, created_at, image_count, last_updated, status
                    FROM users WHERE id = ?
                """, (user_id,))
                row = cursor.fetchone()
                return dict(row) if row else None
        except sqlite3.Error as e:
            logging.error(f"Error getting user: {e}")
            raise

    def get_all_users(self) -> List[Dict[str, Any]]:
        """Get all users
        
        Returns:
            List[Dict[str, Any]]: List of all users
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT id, name, created_at, image_count, last_updated, status
                    FROM users ORDER BY name
                """)
                return [dict(row) for row in cursor.fetchall()]
        except sqlite3.Error as e:
            logging.error(f"Error getting users: {e}")
            raise

    def update_user_image_count(self, user_id: int, count: int) -> None:
        """Update user's image count
        Args:
            user_id (int): User ID
            count (int): Number of images to add to count
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    UPDATE users 
                    SET image_count = image_count + ?,
                        last_updated = CURRENT_TIMESTAMP 
                    WHERE id = ?
                """, (count, user_id))
                conn.commit()
                logging.info(f"Updated image count for user {user_id}: +{count}")
        except sqlite3.Error as e:
            logging.error(f"Error updating user image count: {e}")
            raise

   
    def cleanup_old_data(self, days: int = 30) -> None:
        """Clean up old recognition logs and sessions
        Args:
            days (int): Number of days of data to keep
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    DELETE FROM recognition_logs
                    WHERE timestamp < datetime('now', ?)
                """, (f'-{days} days',))
                cursor.execute("""
                    DELETE FROM recognition_sessions
                    WHERE timestamp < datetime('now', ?)
                    AND status = 'completed'
                """, (f'-{days} days',))
                conn.commit()
                logging.info(f"Cleaned up data older than {days} days")
        except sqlite3.Error as e:
            logging.error(f"Error cleaning up old data: {e}")
            raise

    def save_training_session(self, metrics: Dict[str, Any], num_users: int, total_images: int) -> int:
        """Save training session results
        
        Args:
            metrics (Dict[str, Any]): Training metrics
            num_users (int): Number of users in training
            total_images (int): Total number of images used
            
        Returns:
            int: Training session ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO training_sessions 
                    (num_users, total_images, accuracy, precision_score, 
                     recall_score, f1_score, model_path, training_duration)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    num_users, total_images,
                    metrics['accuracy'], metrics['precision'],
                    metrics['recall'], metrics['f1_score'],
                    metrics.get('model_path', ''),
                    metrics['training_time']
                ))
                session_id = cursor.lastrowid
                conn.commit()
                logging.info(f"Saved training session: {session_id}")
                return session_id
        except sqlite3.Error as e:
            logging.error(f"Error saving training session: {e}")
            raise

    def get_latest_training_metrics(self) -> Optional[Dict[str, Any]]:
        """Get latest training session metrics
        
        Returns:
            Optional[Dict[str, Any]]: Latest training metrics or None
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM training_sessions 
                    ORDER BY timestamp DESC LIMIT 1
                """)
                row = cursor.fetchone()
                return dict(row) if row else None
        except sqlite3.Error as e:
            logging.error(f"Error getting training metrics: {e}")
            raise

    def start_recognition_session(self) -> int:
        """Start new recognition session
        
        Returns:
            int: Recognition session ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO recognition_sessions (status)
                    VALUES ('active')
                """)
                session_id = cursor.lastrowid
                conn.commit()
                logging.info(f"Started recognition session: {session_id}")
                return session_id
        except sqlite3.Error as e:
            logging.error(f"Error starting recognition session: {e}")
            raise
            
    def get_recognition_stats(self, hours: int = 1) -> Dict[str, Any]:
        """Get recognition statistics for the past n hours"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 
                        COUNT(*) as total_recognitions,
                        AVG(confidence) as avg_confidence,
                        COUNT(DISTINCT user_id) as unique_users
                    FROM recognition_logs
                    WHERE timestamp >= datetime('now', ?)
                """, (f'-{hours} hours',))
                
                result = cursor.fetchone()
                return {
                    'total_recognitions': result[0] if result else 0,
                    'avg_confidence': result[1] if result and result[1] is not None else 0,
                    'unique_users': result[2] if result else 0
                }
        except sqlite3.Error as e:
            logger.error(f"Error getting recognition stats: {e}")
            return {'total_recognitions': 0, 'avg_confidence': 0, 'unique_users': 0}

    def log_recognition(self, session_id: int, user_id: int, 
                        confidence: float, frame_path: Optional[str] = None) -> int:
            """Log recognition event
            
            Args:
                session_id (int): Recognition session ID
                user_id (int): Recognized user ID
                confidence (float): Recognition confidence score
                frame_path (Optional[str]): Path to saved frame image
                
            Returns:
                int: Recognition log ID
            """
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        INSERT INTO recognition_logs 
                        (session_id, user_id, confidence, frame_path)
                        VALUES (?, ?, ?, ?)
                    """, (session_id, user_id, confidence, frame_path))
                    log_id = cursor.lastrowid
                    conn.commit()
                    logging.info(f"Logged recognition event: {log_id}")
                    return log_id
            except sqlite3.Error as e:
                logging.error(f"Error logging recognition: {e}")
                raise

    def get_recognition_stats(self, hours: int = 24) -> Dict[str, Any]:
            """Get recognition statistics for the past n hours
            
            Args:
                hours (int): Number of hours to look back
                
            Returns:
                Dict[str, Any]: Recognition statistics
            """
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT 
                            COUNT(*) as total_recognitions,
                            AVG(confidence) as avg_confidence,
                            COUNT(DISTINCT user_id) as unique_users
                        FROM recognition_logs
                        WHERE timestamp >= datetime('now', ?)
                    """, (f'-{hours} hours',))
                    row = cursor.fetchone()
                    return {
                        'total_recognitions': row[0],
                        'avg_confidence': row[1],
                        'unique_users': row[2]
                    }
            except sqlite3.Error as e:
                logging.error(f"Error getting recognition stats: {e}")
                raise

    def close_recognition_session(self, session_id: int) -> None:
            """Close recognition session
            
            Args:
                session_id (int): Recognition session ID
            """
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        UPDATE recognition_sessions
                        SET status = 'completed',
                            duration = (
                                strftime('%s', 'now') - 
                                strftime('%s', timestamp)
                            )
                        WHERE id = ?
                    """, (session_id,))
                    conn.commit()
                    logging.info(f"Closed recognition session: {session_id}")
            except sqlite3.Error as e:
                logging.error(f"Error closing recognition session: {e}")
                raise

    def start_recognition_session(self, session_name: str) -> int:
        """Start new recognition session with name
        
        Args:
            session_name (str): Name for the recognition session
            
        Returns:
            int: Recognition session ID
        """
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                cursor.execute("""
                    INSERT INTO recognition_sessions (session_name, status)
                    VALUES (?, 'active')
                """, (session_name,))
                session_id = cursor.lastrowid
                conn.commit()
                logging.info(f"Started recognition session: {session_id} - {session_name}")
                return session_id
        except sqlite3.Error as e:
            logging.error(f"Error starting recognition session: {e}")
            raise

    def finalize_session_stats(self, session_id: int) -> None:
        """Update final statistics for a recognition session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.cursor()
                
                # Calculate final statistics
                cursor.execute("""
                    UPDATE recognition_sessions
                    SET status = 'completed',
                        duration = (strftime('%s', 'now') - strftime('%s', timestamp)),
                        total_recognitions = (
                            SELECT COUNT(*) 
                            FROM recognition_logs 
                            WHERE session_id = ?
                        ),
                        unique_faces = (
                            SELECT COUNT(DISTINCT user_id) 
                            FROM recognition_logs 
                            WHERE session_id = ?
                        ),
                        avg_confidence = (
                            SELECT AVG(confidence) 
                            FROM recognition_logs 
                            WHERE session_id = ?
                        )
                    WHERE id = ?
                """, (session_id, session_id, session_id, session_id))
                conn.commit()
                logging.info(f"Finalized statistics for session: {session_id}")
        except sqlite3.Error as e:
            logging.error(f"Error finalizing session stats: {e}")
            raise

    def get_session_summary(self, session_id: int) -> Dict[str, Any]:
        """Get summary of recognition session"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.row_factory = sqlite3.Row
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT * FROM session_summaries
                    WHERE id = ?
                """, (session_id,))
                result = cursor.fetchone()
                return dict(result) if result else None
        except sqlite3.Error as e:
            logging.error(f"Error getting session summary: {e}")
            raise
    
    def cleanup_old_data(self, days: int = 30) -> None:
            """Clean up old recognition logs and sessions
            
            Args:
                days (int): Number of days of data to keep
            """
            try:
                with sqlite3.connect(self.db_path) as conn:
                    cursor = conn.cursor()
                    cursor.execute("""
                        DELETE FROM recognition_logs
                        WHERE timestamp < datetime('now', ?)
                    """, (f'-{days} days',))
                    cursor.execute("""
                        DELETE FROM recognition_sessions
                        WHERE timestamp < datetime('now', ?)
                        AND status = 'completed'
                    """, (f'-{days} days',))
                    conn.commit()
                    logging.info(f"Cleaned up data older than {days} days")
            except sqlite3.Error as e:
                logging.error(f"Error cleaning up old data: {e}")
                raise

class LivenessDetector:
    """Handles liveness detection using multiple techniques"""
    def __init__(self):
        self.initialize_models()
        # Keep track of blink states for temporal analysis
        self.eye_blink_history = deque(maxlen=10)
        # Track head movements
        self.head_pose_history = deque(maxlen=20)
        
        # Define thresholds
        self.blink_threshold = 0.3
        self.movement_threshold = 0.15
        self.depth_variation_threshold = 0.2
        
        # Initialize face landmarks detector
        self.face_detector = dlib.get_frontal_face_detector()
        self.landmark_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
        
        logger.info("Liveness detector initialized successfully")

    def initialize_models(self):
        """Initialize deep learning models for liveness detection"""
        try:
            # Base model for feature extraction
            base_model = MobileNetV2(
                input_shape=(224, 224, 3),
                include_top=False,
                weights='imagenet'
            )
            
            # Liveness detection model
            x = base_model.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(1024, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.5)(x)
            predictions = Dense(1, activation='sigmoid')(x)
            
            self.liveness_model = tf.keras.Model(inputs=base_model.input, outputs=predictions)
            
            # Load pre-trained weights if available
            try:
                self.liveness_model.load_weights('liveness_model_weights.h5')
                logger.info("Loaded pre-trained liveness detection weights")
            except:
                logger.warning("No pre-trained weights found for liveness detection")
            
        except Exception as e:
            logger.error(f"Error initializing liveness detection models: {e}")
            raise

    def calculate_eye_aspect_ratio(self, eye_points):
        """Calculate the eye aspect ratio for blink detection"""
        # Compute vertical eye distances
        v1 = distance.euclidean(eye_points[1], eye_points[5])
        v2 = distance.euclidean(eye_points[2], eye_points[4])
        
        # Compute horizontal eye distance
        h = distance.euclidean(eye_points[0], eye_points[3])
        
        # Calculate eye aspect ratio
        ear = (v1 + v2) / (2.0 * h)
        return ear

    def detect_blinks(self, frame, landmarks):
        """Detect natural eye blinks"""
        try:
            # Extract eye regions
            left_eye = np.array([landmarks[36:42]])
            right_eye = np.array([landmarks[42:48]])
            
            # Calculate eye aspect ratios
            left_ear = self.calculate_eye_aspect_ratio(left_eye[0])
            right_ear = self.calculate_eye_aspect_ratio(right_eye[0])
            
            # Average eye aspect ratio
            ear = (left_ear + right_ear) / 2.0
            self.eye_blink_history.append(ear)
            
            # Detect blink patterns
            if len(self.eye_blink_history) >= 3:
                # Look for natural blink pattern (quick down, quick up)
                if (self.eye_blink_history[-2] < self.blink_threshold and
                    self.eye_blink_history[-1] > self.blink_threshold and
                    self.eye_blink_history[-3] > self.blink_threshold):
                    return True
                    
            return False
            
        except Exception as e:
            logger.error(f"Error in blink detection: {e}")
            return False

    def analyze_depth(self, frame):
        """Analyze depth variations in the image"""
        try:
            # Convert to grayscale for depth analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Apply Sobel operators for depth estimation
            sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
            sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
            
            # Calculate gradient magnitude
            gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
            
            # Analyze depth variations
            depth_variation = np.std(gradient_magnitude) / np.mean(gradient_magnitude)
            
            # Real faces typically show more depth variation
            return depth_variation > self.depth_variation_threshold
            
        except Exception as e:
            logger.error(f"Error in depth analysis: {e}")
            return False

    def detect_head_movement(self, landmarks):
        """Detect natural head movements"""
        try:
            # Calculate head pose using facial landmarks
            nose_tip = landmarks[30]
            left_eye = np.mean(landmarks[36:42], axis=0)
            right_eye = np.mean(landmarks[42:48], axis=0)
            
            # Calculate head orientation
            eye_center = (left_eye + right_eye) / 2
            head_vector = nose_tip - eye_center
            
            self.head_pose_history.append(head_vector)
            
            if len(self.head_pose_history) >= 10:
                # Analyze movement patterns
                movement = np.std([np.linalg.norm(v - self.head_pose_history[-1]) 
                                 for v in self.head_pose_history])
                return movement > self.movement_threshold
                
            return False
            
        except Exception as e:
            logger.error(f"Error in head movement detection: {e}")
            return False

    def analyze_texture_patterns(self, face_region):
        """Analyze texture patterns to detect printed faces"""
        try:
            # Convert to grayscale
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            
            # Apply LBP (Local Binary Patterns)
            radius = 1
            n_points = 8 * radius
            lbp = local_binary_pattern(gray, n_points, radius, method='uniform')
            
            # Calculate histogram of patterns
            n_bins = int(lbp.max() + 1)
            hist, _ = np.histogram(lbp, density=True, bins=n_bins, range=(0, n_bins))
            
            # Real faces typically have more texture variation
            texture_variation = np.std(hist)
            return texture_variation > 0.1
            
        except Exception as e:
            logger.error(f"Error in texture analysis: {e}")
            return False

    def check_liveness(self, frame, face_location):
        """Comprehensive liveness check combining multiple techniques"""
        try:
            x, y, w, h = face_location
            face_region = frame[y:y+h, x:x+w]
            
            # Get facial landmarks
            rect = dlib.rectangle(x, y, x+w, y+h)
            landmarks = self.landmark_predictor(frame, rect)
            landmarks = np.array([[p.x, p.y] for p in landmarks.parts()])
            
            # Perform multiple liveness checks
            liveness_scores = {
                'blink_detection': self.detect_blinks(frame, landmarks),
                'depth_analysis': self.analyze_depth(face_region),
                'head_movement': self.detect_head_movement(landmarks),
                'texture_analysis': self.analyze_texture_patterns(face_region)
            }
            
            # Deep learning-based liveness detection
            face_input = cv2.resize(face_region, (224, 224))
            face_input = tf.keras.applications.mobilenet_v2.preprocess_input(face_input)
            face_input = np.expand_dims(face_input, axis=0)
            liveness_prediction = self.liveness_model.predict(face_input)[0][0]
            
            # Combine all checks for final decision
            is_live = (
                liveness_prediction > 0.5 and
                sum(liveness_scores.values()) >= 2  # At least 2 passive checks must pass
            )
            
            return is_live, liveness_scores, liveness_prediction
            
        except Exception as e:
            logger.error(f"Error in liveness detection: {e}")
            return False, {}, 0.0

class FaceRecognitionSystem:
    """Main face recognition system class"""
    def __init__(self):
        """Initialize the face recognition system"""
        self.setup_directories()
        self.config = ConfigManager()
        self.db = DatabaseManager(self.base_dir / 'face_recognition.db')
        self.encryption_manager = EncryptionManager(self.base_dir / 'encryption_key.key')
        self.initialize_models()
        self.load_classifier()
        logger.info("Face Recognition System initialized successfully")

    def setup_directories(self) -> None:
        """Set up necessary directories"""
        self.base_dir = Path('face_recognition_data')
        self.model_dir = self.base_dir / 'models'
        self.image_dir = self.base_dir / 'images'
        self.results_dir = self.base_dir / 'results'
        self.logs_dir = self.base_dir / 'logs'
        
        for directory in [self.base_dir, self.model_dir, self.image_dir, 
                         self.results_dir, self.logs_dir]:
            directory.mkdir(parents=True, exist_ok=True)

    def initialize_models(self) -> None:
        """Initialize CV and ML models"""
        try:
            # Load face detection model
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            if not os.path.exists(cascade_path):
                raise FileNotFoundError(f"Cascade classifier not found at {cascade_path}")
            
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            if self.face_cascade.empty():
                raise RuntimeError("Failed to load cascade classifier")
            
            # Initialize MobileNetV2 for feature extraction
            self.mobilenet = MobileNetV2(
                weights='imagenet',
                include_top=False,
                input_shape=(224, 224, 3),
                pooling='avg'
            )
            
            logger.info("Models initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing models: {e}")
            raise

    def load_classifier(self) -> None:
        """Load the trained classifier if it exists"""
        try:
            model_files = list(self.model_dir.glob('*.pkl'))
            if model_files:
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                with open(latest_model, 'rb') as f:
                    self.classifier = pickle.load(f)
                logger.info(f"Loaded classifier from {latest_model}")
            else:
                self.classifier = None
                logger.info("No trained classifier found")
        except Exception as e:
            logger.error(f"Error loading classifier: {e}")
            self.classifier = None

    def save_model(self, metrics: Dict[str, Any]) -> str:
        """Save trained model and metrics"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        model_path = self.model_dir / f"model_{timestamp}.pkl"
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(self.classifier, f)
            
            # Save metrics separately
            metrics_path = self.results_dir / f"metrics_{timestamp}.json"
            with open(metrics_path, 'w') as f:
                json.dump(metrics, f, indent=4)
            
            return str(model_path)
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            raise

    def detect_face(self, frame: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[tuple]]:
        """Detect and preprocess face from frame"""
        try:
            if frame is None:
                return None, None

            # Convert to grayscale
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Enhance image
            gray = cv2.equalizeHist(gray)
            
            # Detect faces with different scales
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=self.config.config['min_face_size'],
                flags=cv2.CASCADE_SCALE_IMAGE
            )
            
            if len(faces) == 0:
                return None, None
            
            # Get the largest face
            faces = sorted(faces, key=lambda x: x[2] * x[3], reverse=True)
            (x, y, w, h) = faces[0]
            
            # Add padding
            padding = int(0.1 * w)
            x = max(0, x - padding)
            y = max(0, y - padding)
            w = min(frame.shape[1] - x, w + 2 * padding)
            h = min(frame.shape[0] - y, h + 2 * padding)
            
            # Extract and preprocess face region
            face = frame[y:y+h, x:x+w]
            face = cv2.resize(face, self.config.config['feature_extraction']['image_size'])
            
            # Apply image enhancements
            face = cv2.convertScaleAbs(face, alpha=1.1, beta=10)
            
            return face, (x, y, w, h)

        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return None, None

    def extract_features(self, face: np.ndarray) -> Optional[np.ndarray]:
        """Extract features using MobileNetV2"""
        try:
            if face is None:
                return None

            # Preprocess image for MobileNetV2
            face = cv2.resize(face, self.config.config['feature_extraction']['image_size'])
            face = keras_image.img_to_array(face)
            face = np.expand_dims(face, axis=0)
            face = tf.keras.applications.mobilenet_v2.preprocess_input(face)
            
            # Extract features
            features = self.mobilenet.predict(face, verbose=0)
            features = features.flatten()
            
            # Normalize features
            features = features / np.linalg.norm(features)
            
            return features

        except Exception as e:
            logger.error(f"Error extracting features: {e}")
            return None

    def capture_dataset(self, username: str, num_images: int = 20, 
                       progress_callback: Optional[callable] = None) -> List[Path]:
        """Capture dataset for a user with progress tracking"""
        image_paths = []
        try:
            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                raise RuntimeError("Failed to open camera")

            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.config['camera']['frame_size'][0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.config['camera']['frame_size'][0])

            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.config['camera']['frame_size'][0])
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.config['camera']['frame_size'][1])

            # Add user to database
            user_id = self.db.add_user(username)

            captured_count = 0
            while captured_count < num_images:
                ret, frame = cap.read()
                if not ret:
                    raise RuntimeError("Failed to capture frame")

                face, bbox = self.detect_face(frame)
                if face is not None:
                    # Save face image with timestamp
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                    image_path = self.image_dir / f"{username}_{user_id}_{timestamp}.jpg"
                    cv2.imwrite(str(image_path), face)
                    image_paths.append(image_path)
                    captured_count += 1
                    
                    # Update progress
                    if progress_callback:
                        progress_callback(captured_count, num_images)
                    
                    time.sleep(self.config.config['camera']['capture_interval'])

            # Update user's image count
            self.db.update_user_image_count(user_id, len(image_paths))

            logger.info(f"Successfully captured {len(image_paths)} images for user {username}")
            return image_paths

        except Exception as e:
            logger.error(f"Error capturing dataset: {e}")
            # Cleanup partial captures
            for path in image_paths:
                try:
                    if isinstance(path, Path):
                        path.unlink(missing_ok=True)
                except Exception:
                    pass
            raise
        finally:
            if 'cap' in locals():
                cap.release()

    def train_model(self, progress_callback: Optional[callable] = None) -> Dict[str, Any]:
        """Train face recognition model with encrypted images"""
        start_time = time.time()
        try:
            features = []
            labels = []
            user_images = {}

            # Get total image count for progress tracking
            image_files = list(self.image_dir.glob("*.jpg"))
            total_images = len(image_files)

            if total_images == 0:
                raise ValueError("No training images found")

            for idx, img_path in enumerate(image_files):
                try:
                    # Load and decrypt image
                    image = self.encryption_manager.secure_load_image(img_path)
                    
                    # Extract encrypted filename info
                    encrypted_name = img_path.stem
                    decrypted_name = self.encryption_manager.decrypt_image(
                        b64decode(encrypted_name)
                    ).tobytes().decode('utf-8')
                    username, user_id, _ = decrypted_name.split('_')
                    user_id = int(user_id)
                    
                    if image is not None:
                        face_features = self.extract_features(image)
                        if face_features is not None:
                            features.append(face_features)
                            labels.append(user_id)
                            user_images[user_id] = user_images.get(user_id, 0) + 1
                    
                    if progress_callback:
                        progress_callback(idx + 1, total_images)

                except Exception as e:
                    logging.error(f"Error processing image {img_path}: {e}")
                    continue


            # Convert to numpy arrays and train
            X = np.array(features)
            y = np.array(labels)

            # Split data
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.config.config['training']['test_size'],
                random_state=self.config.config['training']['random_state']
            )

            # Train classifier
            self.classifier = SVC(kernel='rbf', probability=True, class_weight='balanced')
            self.classifier.fit(X_train, y_train)

            # Calculate metrics
            y_pred = self.classifier.predict(X_test)
            metrics = {
                'accuracy': float(accuracy_score(y_test, y_pred)),
                'precision': float(precision_score(y_test, y_pred, average='weighted')),
                'recall': float(recall_score(y_test, y_pred, average='weighted')),
                'f1_score': float(f1_score(y_test, y_pred, average='weighted')),
                'training_time': float(time.time() - start_time)
            }

            # Save confusion matrix plot
            cm = confusion_matrix(y_test, y_pred)
            plt.figure(figsize=(10, 8))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Confusion Matrix')
            plt.ylabel('True Label')
            plt.xlabel('Predicted Label')
            cm_path = self.results_dir / f"confusion_matrix_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(cm_path)
            plt.close()

            # Save model and update database
            model_path = self.save_model(metrics)
            metrics['model_path'] = model_path
            self.db.save_training_session(metrics, len(user_images), total_images)

            return metrics

        except Exception as e:
            logger.error(f"Error training model: {e}")
            raise

    def recognize_face(self, frame: np.ndarray) -> Tuple[Optional[int], Optional[float], Optional[tuple]]:
        """Recognize face in frame"""
        try:
            if self.classifier is None:
                return None, None, None

            face, bbox = self.detect_face(frame)
            if face is not None:
                features = self.extract_features(face)
                if features is not None:
                    # Get prediction and confidence
                    prediction = self.classifier.predict([features])[0]
                    confidence = self.classifier.predict_proba([features]).max()
                    
                    return prediction, confidence, bbox

            return None, None, None

        except Exception as e:
            logger.error(f"Error in face recognition: {e}")
            return None, None, None

def show_dataset_collection():
    """Dataset collection interface"""
    st.title("Dataset Collection")
    
    # User input form
    with st.form("dataset_form"):
        username = st.text_input("Username", 
            help="Enter name of the person to capture")
        num_images = st.slider("Number of images", 
            min_value=10, max_value=50, value=20,
            help="More images generally lead to better recognition")
        submitted = st.form_submit_button("Start Capture")

    if submitted:
        if not username or not username.strip():
            st.error("Please enter a valid username")
            return

        try:
            progress_bar = st.progress(0)
            status_text = st.empty()

            def update_progress(current, total):
                progress = float(current) / float(total)
                progress_bar.progress(progress)
                status_text.text(f"Capturing image {current}/{total}")

            image_paths = st.session_state.system.capture_dataset(
                username.strip(), 
                num_images,
                progress_callback=update_progress
            )

            if image_paths:
                st.success(f"Successfully captured {len(image_paths)} images")
                
                # Display sample images
                st.subheader("Sample Images")
                cols = st.columns(min(5, len(image_paths)))
                for idx, path in enumerate(image_paths[:5]):
                    cols[idx].image(str(path))
            else:
                st.warning("No images were captured. Please try again.")

        except Exception as e:
            logger.error(f"Error in dataset collection: {e}")
            st.error(f"Error during image capture: {str(e)}")

def show_model_training():
    """Model training interface"""
    st.title("Model Training")

    try:
        # Check user count
        with sqlite3.connect(st.session_state.system.db.db_path) as conn:
            cursor = conn.cursor()
            cursor.execute("SELECT COUNT(DISTINCT id) FROM users")
            user_count = cursor.fetchone()[0]

        if user_count < 2:
            st.warning("At least 2 different users are required for training. Please collect more data.")
            return

        col1, col2 = st.columns(2)
        with col1:
            if st.button("Train Model"):
                progress_bar = st.progress(0)
                status_text = st.empty()

                def update_progress(current, total):
                    progress = float(current) / float(total)
                    progress_bar.progress(progress)
                    status_text.text(f"Processing image {current}/{total}")

                with st.spinner("Training model... This may take a few minutes."):
                    metrics = st.session_state.system.train_model(
                        progress_callback=update_progress
                    )

                    # Display metrics
                    st.success("Model training completed!")
                    
                    metrics_cols = st.columns(4)
                    metrics_cols[0].metric("Accuracy", f"{metrics['accuracy']:.2%}")
                    metrics_cols[1].metric("Precision", f"{metrics['precision']:.2%}")
                    metrics_cols[2].metric("Recall", f"{metrics['recall']:.2%}")
                    metrics_cols[3].metric("F1 Score", f"{metrics['f1_score']:.2%}")
                    
                    st.info(f"Training time: {metrics['training_time']:.2f} seconds")

                    # Show confusion matrix
                    if (st.session_state.system.results_dir / "confusion_matrix_latest.png").exists():
                        st.image(str(st.session_state.system.results_dir / "confusion_matrix_latest.png"))

        with col2:
            # Display training history
            st.subheader("Training History")
            with sqlite3.connect(st.session_state.system.db.db_path) as conn:
                history = pd.read_sql("""
                    SELECT timestamp, accuracy, total_images, training_duration
                    FROM training_sessions
                    ORDER BY timestamp DESC
                    LIMIT 5
                """, conn)
                
                if not history.empty:
                    st.dataframe(history)
                else:
                    st.info("No training history available")

    except Exception as e:
        logger.error(f"Error in model training: {e}")
        st.error(f"Error during training: {str(e)}")

def show_recognition():
    """Face recognition interface with session naming"""
    st.title("Face Recognition")

    if not hasattr(st.session_state.system, 'classifier') or st.session_state.system.classifier is None:
        st.warning("No trained model available. Please train the model first.")
        return

    try:
        # Create columns for the interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Session name input
            if not st.session_state.recognition_active:
                session_name = st.text_input(
                    "Session Name",
                    help="Enter a name for this recognition session",
                    key="session_name"
                )
            
            # Camera feed placeholder
            video_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Control buttons
            button_col1, button_col2 = st.columns(2)
            
            start_button = button_col1.button(
                "Start Recognition",
                disabled=st.session_state.recognition_active or 
                         (not st.session_state.get('session_name', '').strip())
            )
            
            stop_button = button_col2.button(
                "Stop Recognition",
                disabled=not st.session_state.recognition_active
            )

            if start_button and st.session_state.get('session_name'):
                st.session_state.recognition_active = True
                st.session_state.recognition_session_id = st.session_state.system.db.start_recognition_session(
                    st.session_state.session_name
                )
                
            if stop_button:
                st.session_state.recognition_active = False
                if hasattr(st.session_state, 'recognition_session_id'):
                    # Finalize session statistics
                    st.session_state.system.db.finalize_session_stats(
                        st.session_state.recognition_session_id
                    )
                    
                    # Show session summary
                    summary = st.session_state.system.db.get_session_summary(
                        st.session_state.recognition_session_id
                    )
                    if summary:
                        st.success("Recognition Session Completed!")
                        st.write("Session Summary:")
                        st.write(f"- Session Name: {summary['session_name']}")
                        st.write(f"- Total Recognitions: {summary['total_recognitions']}")
                        st.write(f"- Unique Faces: {summary['unique_faces']}")
                        st.write(f"- Average Confidence: {summary['avg_confidence']:.1%}")
                        st.write(f"- Recognized People: {summary['recognized_people']}")

        with col2:
            # Recognition Logs Section
            st.subheader("Recognition Logs")
            
            if st.session_state.recognition_active:
                # Get current session logs
                with sqlite3.connect(st.session_state.system.db.db_path) as conn:
                    logs_df = pd.read_sql("""
                        SELECT 
                            r.timestamp,
                            u.name as recognized_person,
                            r.confidence
                        FROM recognition_logs r
                        JOIN users u ON r.user_id = u.id
                        WHERE r.session_id = ?
                        ORDER BY r.timestamp DESC
                        LIMIT 10
                    """, conn, params=(st.session_state.recognition_session_id,))
                    
                    if not logs_df.empty:
                        # Format the dataframe for display
                        logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp']).dt.strftime('%H:%M:%S')
                        logs_df['confidence'] = logs_df['confidence'].apply(lambda x: f"{x:.1%}")
                        st.dataframe(logs_df, use_container_width=True)
                    else:
                        st.info("No recognitions in current session")
            
            # Statistics
            st.subheader("Current Session Statistics")
            if hasattr(st.session_state, 'recognition_session_id'):
                with sqlite3.connect(st.session_state.system.db.db_path) as conn:
                    stats = pd.read_sql("""
                        SELECT 
                            COUNT(*) as total_recognitions,
                            COUNT(DISTINCT user_id) as unique_faces,
                            AVG(confidence) as avg_confidence
                        FROM recognition_logs
                        WHERE session_id = ?
                    """, conn, params=(st.session_state.recognition_session_id,))
                    
                    if not stats.empty:
                        st.metric("Total Recognitions", stats['total_recognitions'].iloc[0])
                        st.metric("Unique Faces", stats['unique_faces'].iloc[0])
                        if stats['avg_confidence'].iloc[0]:
                            st.metric("Average Confidence", f"{stats['avg_confidence'].iloc[0]:.1%}")

        if st.session_state.recognition_active:
            cap = cv2.VideoCapture(0)
            try:
                while st.session_state.recognition_active:
                    ret, frame = cap.read()
                    if not ret:
                        break

                    # Recognize face
                    user_id, confidence, bbox = st.session_state.system.recognize_face(frame)
                    
                    if user_id is not None and bbox is not None:
                        x, y, w, h = bbox
                        color = (0, 255, 0) if confidence > 0.5 else (0, 0, 255)
                        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
                        
                        try:
                            # Get username for recognized user_id
                            with sqlite3.connect(st.session_state.system.db.db_path) as conn:
                                cursor = conn.cursor()
                                cursor.execute("SELECT name FROM users WHERE id = ?", (user_id,))
                                result = cursor.fetchone()
                                username = result[0] if result else "Unknown"

                            if confidence > 0.3:
                                label = f"{username} ({confidence:.2%})"
                                status_placeholder.success(f"Recognized: {label}")
                                
                                # Save frame and log recognition
                                frame_path = str(st.session_state.system.logs_dir / 
                                              f"recognition_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")
                                cv2.imwrite(frame_path, frame)
                                
                                st.session_state.system.db.log_recognition(
                                    st.session_state.recognition_session_id,
                                    user_id,
                                    confidence,
                                    frame_path
                                )
                            else:
                                label = "Unknown"
                                status_placeholder.warning("Unknown face detected")
                            
                            cv2.putText(frame, label, (x, y-10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                        except sqlite3.Error as e:
                            logger.error(f"Database error during recognition: {e}")
                            continue

                    # Display frame
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(frame_rgb, channels="RGB")

                    # Short sleep to prevent high CPU usage
                    time.sleep(0.1)

            finally:
                cap.release()
                status_placeholder.empty()
                video_placeholder.empty()

    except Exception as e:
        logger.error(f"Error in face recognition: {e}")
        st.error(f"Error during recognition: {str(e)}")

def show_settings():
    """Settings interface"""
    st.title("System Settings")
    
    try:
        config = st.session_state.system.config.config
        
        st.subheader("Detection Settings")
        min_face_size = st.slider(
            "Minimum Face Size",
            min_value=20,
            max_value=100,
            value=config['min_face_size'][0],
            help="Minimum size of face to detect (pixels)"
        )
        
        detection_confidence = st.slider(
            "Detection Confidence",
            min_value=0.1,
            max_value=1.0,
            value=config['detection_confidence'],
            help="Minimum confidence threshold for face detection"
        )
        
        st.subheader("Camera Settings")
        capture_interval = st.slider(
            "Capture Interval",
            min_value=0.1,
            max_value=1.0,
            value=config['camera']['capture_interval'],
            help="Time between captures (seconds)"
        )
        
        frame_width = st.number_input(
            "Frame Width",
            min_value=320,
            max_value=1920,
            value=config['camera']['frame_size'][0],
            help="Camera frame width"
        )
        
        frame_height = st.number_input(
            "Frame Height",
            min_value=240,
            max_value=1080,
            value=config['camera']['frame_size'][1],
            help="Camera frame height"
        )
        
        if st.button("Save Settings"):
            # Update configuration
            config['min_face_size'] = (min_face_size, min_face_size)
            config['detection_confidence'] = detection_confidence
            config['camera']['capture_interval'] = capture_interval
            config['camera']['frame_size'] = (frame_width, frame_height)
            
            # Save configuration
            st.session_state.system.config.save_config()
            st.success("Settings saved successfully!")
            
    except Exception as e:
        logger.error(f"Error in settings: {e}")
        st.error(f"Error updating settings: {str(e)}")

def show_system_logs():
    """System logs interface"""
    st.title("System Logs")
    
    try:
        # Read last 100 lines from log file
        log_file = Path('face_recognition.log')
        if log_file.exists():
            with open(log_file, 'r') as f:
                lines = f.readlines()[-100:]
                
            st.code(''.join(lines), language='text')
        else:
            st.info("No logs available")
            
    except Exception as e:
        logger.error(f"Error reading logs: {e}")
        st.error(f"Error displaying logs: {str(e)}")



def init_session_state():
    """Initialize session state variables"""
    # Initialize authentication-related state
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if 'current_user' not in st.session_state:
        st.session_state.current_user = None
        
    # Initialize system-related state
    if 'system_initialized' not in st.session_state:
        try:
            st.session_state.system = FaceRecognitionSystem()
            st.session_state.system_initialized = True
            st.session_state.camera_active = False
            st.session_state.recognition_active = False
            st.session_state.recognition_session_id = None
            logger.info("Session state initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing session state: {e}")
            st.error(f"Error initializing system: {str(e)}")

def main():
    """Main application entry point"""
    st.set_page_config(
        page_title="Face Recognition System",
        page_icon="👤",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    # Initialize session state
    init_session_state()

    # Authentication check
    if not st.session_state.authenticated:
        authenticate()
        return
    
    # Sidebar navigation
    st.sidebar.title("Face Recognition System")
    
    menu_options = {
        "Dashboard": show_dashboard,
        "Dataset Collection": show_dataset_collection,
        "Model Training": show_model_training,
        "Face Recognition": show_recognition,
        "Settings": show_settings,
        "System Logs": show_system_logs
    }
    
    selection = st.sidebar.radio("Navigation", list(menu_options.keys()))
    
    # System status and user info
    if st.session_state.current_user:
        st.sidebar.success(f"Logged in as: {st.session_state.current_user['name']}")
        if st.sidebar.button("Logout"):
            st.session_state.authenticated = False
            st.session_state.current_user = None
            st.experimental_rerun()
    
    # Display version information
    st.sidebar.info(f"""
        System Version: 1.0.0
        Last Updated: {datetime.now().strftime('%Y-%m-%d')}
    """)

    # Display selected page
    try:
        menu_options[selection]()
    except Exception as e:
        logger.error(f"Error displaying page {selection}: {e}")
        st.error("An error occurred. Please try again.")

    # Memory cleanup
    gc.collect()

def authenticate():
    """Authenticate user"""
    st.title("Login")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        username = st.text_input("Username", key="username")
        password = st.text_input("Password", type="password", key="password")
        
        if st.button("Login"):
            try:
                user = st.session_state.system.db.authenticate_user(username, password)
                if user:
                    st.session_state.authenticated = True
                    st.session_state.current_user = user
                    st.success("Login successful!")
                    st.experimental_rerun()
                else:
                    st.error("Invalid username or password")
            except Exception as e:
                logger.error(f"Authentication error: {e}")
                st.error("An error occurred during authentication")

def show_dashboard():
    """Display enhanced dashboard with recognition statistics"""
    st.title("Face Recognition Dashboard")
    
    try:
        # User stats
        st.subheader("System Overview")
        col1, col2, col3, col4 = st.columns(4)
        
        with sqlite3.connect(st.session_state.system.db.db_path) as conn:
            # System stats
            users_df = pd.read_sql("""
                SELECT COUNT(*) as total_users, 
                    SUM(image_count) as total_images
                FROM users
            """, conn)
            
            recognition_df = pd.read_sql("""
                SELECT COUNT(*) as total_recognitions,
                    COUNT(DISTINCT user_id) as unique_users,
                    AVG(confidence) as avg_confidence
                FROM recognition_logs
                WHERE timestamp >= datetime('now', '-24 hours')
            """, conn)
            
            # Display metrics
            col1.metric("Registered Users", int(users_df['total_users'].iloc[0]))
            col2.metric("Total Images", int(users_df['total_images'].iloc[0]))
            col3.metric("24h Recognitions", int(recognition_df['total_recognitions'].iloc[0]))
            col4.metric("Recognition Accuracy", 
                       f"{float(recognition_df['avg_confidence'].iloc[0])*100:.1f}%")
        
        # Recognition Statistics
        st.subheader("Recognition Statistics (Last 24 Hours)")
        
        # Get detailed recognition stats by user
        recognition_stats = st.session_state.system.db.get_recognition_stats_by_user()
        if not recognition_stats.empty:
            # Format the dataframe
            recognition_stats['avg_confidence'] = recognition_stats['avg_confidence'].apply(
                lambda x: f"{x*100:.1f}%"
            )
            recognition_stats['first_recognition'] = pd.to_datetime(
                recognition_stats['first_recognition']
            ).dt.strftime('%Y-%m-%d %H:%M')
            recognition_stats['last_recognition'] = pd.to_datetime(
                recognition_stats['last_recognition']
            ).dt.strftime('%Y-%m-%d %H:%M')
            
            st.dataframe(recognition_stats, use_container_width=True)
        else:
            st.info("No recognition data available for the last 24 hours")
        
        # Recent Activity
        st.subheader("Recent Recognition Activity")
        with sqlite3.connect(st.session_state.system.db.db_path) as conn:
            recent_activity = pd.read_sql("""
                SELECT 
                    u.name as recognized_person,
                    r.timestamp,
                    r.confidence,
                    r.frame_path
                FROM recognition_logs r
                JOIN users u ON r.user_id = u.id
                ORDER BY r.timestamp DESC
                LIMIT 10
            """, conn)
            
            if not recent_activity.empty:
                # Format the dataframe
                recent_activity['confidence'] = recent_activity['confidence'].apply(
                    lambda x: f"{x*100:.1f}%"
                )
                recent_activity['timestamp'] = pd.to_datetime(
                    recent_activity['timestamp']
                ).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                st.dataframe(recent_activity, use_container_width=True)
            else:
                st.info("No recent activity")
        
        # Training History
        st.subheader("Model Training History")
        with sqlite3.connect(st.session_state.system.db.db_path) as conn:
            training_history = pd.read_sql("""
                SELECT 
                    timestamp,
                    accuracy,
                    precision_score,
                    recall_score,
                    f1_score,
                    num_users,
                    total_images,
                    training_duration
                FROM training_sessions
                ORDER BY timestamp DESC
                LIMIT 5
            """, conn)
            
            if not training_history.empty:
                # Format metrics as percentages
                for col in ['accuracy', 'precision_score', 'recall_score', 'f1_score']:
                    training_history[col] = training_history[col].apply(
                        lambda x: f"{x*100:.1f}%"
                    )
                training_history['training_duration'] = training_history[
                    'training_duration'
                ].apply(lambda x: f"{x:.1f}s")
                
                st.dataframe(training_history, use_container_width=True)
            else:
                st.info("No training history available")

    except Exception as e:
        logger.error(f"Error displaying dashboard: {e}")
        st.error("Error loading dashboard data")

def logout():
    """Logout user"""
    st.session_state.authenticated = False
    st.session_state.current_user = None
    st.experimental_rerun()


if __name__ == "__main__":
    main()




    # def show_recognition():
    # """Face recognition interface with comprehensive session tracking"""
    # st.title("Face Recognition")

    # if not hasattr(st.session_state.system, 'classifier') or st.session_state.system.classifier is None:
    #     st.warning("No trained model available. Please train the model first.")
    #     return

    # try:
    #     col1, col2 = st.columns([2, 1])
        
    #     with col1:
    #         # Session name input before starting
    #         if not st.session_state.recognition_active:
    #             session_name = st.text_input(
    #                 "Session Name",
    #                 help="Enter a name for this recognition session",
    #                 key="session_name",
    #                 placeholder="e.g., Morning Class, Team Meeting"
    #             )
            
    #         video_placeholder = st.empty()
    #         status_placeholder = st.empty()
            
    #         button_col1, button_col2 = st.columns(2)
            
    #         start_button = button_col1.button(
    #             "Start Recognition",
    #             disabled=st.session_state.recognition_active or 
    #                      (not st.session_state.get('session_name', '').strip())
    #         )
            
    #         stop_button = button_col2.button(
    #             "Stop Recognition",
    #             disabled=not st.session_state.recognition_active
    #         )

    #         if start_button and st.session_state.get('session_name'):
    #             st.session_state.recognition_active = True
    #             st.session_state.recognition_session_id = st.session_state.system.db.start_recognition_session(
    #                 st.session_state.session_name
    #             )
                
    #         if stop_button:
    #             st.session_state.recognition_active = False
    #             if hasattr(st.session_state, 'recognition_session_id'):
    #                 st.session_state.system.db.finalize_session_stats(
    #                     st.session_state.recognition_session_id
    #                 )

    #     with col2:
    #         # Session History
    #         st.subheader("Recognition Sessions History")
    #         with sqlite3.connect(st.session_state.system.db.db_path) as conn:
    #             sessions_history = pd.read_sql("""
    #                 SELECT 
    #                     s.session_name,
    #                     s.timestamp,
    #                     s.total_recognitions,
    #                     s.unique_faces,
    #                     s.avg_confidence,
    #                     GROUP_CONCAT(DISTINCT u.name) as recognized_people
    #                 FROM recognition_sessions s
    #                 LEFT JOIN recognition_logs l ON s.id = l.session_id
    #                 LEFT JOIN users u ON l.user_id = u.id
    #                 GROUP BY s.id
    #                 ORDER BY s.timestamp DESC
    #                 LIMIT 5
    #             """, conn)
                
    #             if not sessions_history.empty:
    #                 sessions_history['timestamp'] = pd.to_datetime(
    #                     sessions_history['timestamp']
    #                 ).dt.strftime('%Y-%m-%d %H:%M')
    #                 sessions_history['avg_confidence'] = sessions_history[
    #                     'avg_confidence'
    #                 ].apply(lambda x: f"{x*100:.1f}%" if x else "N/A")
    #                 st.dataframe(sessions_history, use_container_width=True)
    #             else:
    #                 st.info("No recognition sessions recorded yet")
            
    #         # Current Session Stats
    #         if st.session_state.recognition_active:
    #             st.subheader("Current Session")
    #             with sqlite3.connect(st.session_state.system.db.db_path) as conn:
    #                 current_stats = pd.read_sql("""
    #                     SELECT 
    #                         COUNT(*) as total_recognitions,
    #                         COUNT(DISTINCT user_id) as unique_faces,
    #                         AVG(confidence) as avg_confidence,
    #                         GROUP_CONCAT(DISTINCT u.name) as recognized_people
    #                     FROM recognition_logs l
    #                     JOIN users u ON l.user_id = u.id
    #                     WHERE session_id = ?
    #                 """, conn, params=(st.session_state.recognition_session_id,))
                    
    #                 if not current_stats.empty:
    #                     st.metric("Total Recognitions", current_stats['total_recognitions'].iloc[0])
    #                     st.metric("Unique Faces", current_stats['unique_faces'].iloc[0])
    #                     if current_stats['avg_confidence'].iloc[0]:
    #                         st.metric("Average Confidence", 
    #                                 f"{current_stats['avg_confidence'].iloc[0]*100:.1f}%")
    #                     if current_stats['recognized_people'].iloc[0]:
    #                         st.write("Recognized People:")
    #                         st.write(current_stats['recognized_people'].iloc[0])

    #     if st.session_state.recognition_active:
    #         cap = cv2.VideoCapture(0)
    #         try:
    #             while st.session_state.recognition_active:
    #                 ret, frame = cap.read()
    #                 if not ret:
    #                     break

    #                 # Enhance face recognition logic
    #                 user_id, confidence, bbox = st.session_state.system.recognize_face(frame)
                    
    #                 if user_id is not None and bbox is not None:
    #                     x, y, w, h = bbox
                        
    #                     try:
    #                         # Get username and determine recognition confidence
    #                         with sqlite3.connect(st.session_state.system.db.db_path) as conn:
    #                             cursor = conn.cursor()
    #                             cursor.execute("SELECT name FROM users WHERE id = ?", (user_id,))
    #                             result = cursor.fetchone()
    #                             username = result[0] if result else "Unknown"

    #                         # Only recognize if confidence is high enough
    #                         if confidence > 0.5:  # Adjust threshold as needed
    #                             color = (0, 255, 0)  # Green for high confidence
    #                             label = f"{username} ({confidence:.2%})"
    #                             status_placeholder.success(f"Recognized: {label}")
                                
    #                             # Save frame and log recognition
    #                             frame_path = str(st.session_state.system.logs_dir / 
    #                                           f"recognition_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}.jpg")
    #                             cv2.imwrite(frame_path, frame)
                                
    #                             # Update recognition session with user_id
    #                             cursor.execute("""
    #                                 UPDATE recognition_sessions 
    #                                 SET user_id = ?
    #                                 WHERE id = ?
    #                             """, (user_id, st.session_state.recognition_session_id))
                                
    #                             # Log the recognition event
    #                             st.session_state.system.db.log_recognition(
    #                                 st.session_state.recognition_session_id,
    #                                 user_id,
    #                                 confidence,
    #                                 frame_path
    #                             )
    #                             conn.commit()
    #                         else:
    #                             color = (0, 0, 255)  # Red for low confidence
    #                             label = "Unknown"
    #                             status_placeholder.warning("Low confidence detection")
                            
    #                         # Draw rectangle and label
    #                         cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
    #                         cv2.putText(frame, label, (x, y-10), 
    #                                   cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

    #                     except sqlite3.Error as e:
    #                         logger.error(f"Database error during recognition: {e}")
    #                         continue

    #                 # Display frame
    #                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    #                 video_placeholder.image(frame_rgb, channels="RGB")
    #                 time.sleep(0.1)

    #         finally:
    #             cap.release()
                
    #             # Show session summary when stopped
    #             if not st.session_state.recognition_active:
    #                 with sqlite3.connect(st.session_state.system.db.db_path) as conn:
    #                     summary = pd.read_sql("""
    #                         SELECT 
    #                             s.session_name,
    #                             s.timestamp,
    #                             COUNT(DISTINCT l.user_id) as unique_faces,
    #                             COUNT(*) as total_recognitions,
    #                             AVG(l.confidence) as avg_confidence,
    #                             GROUP_CONCAT(DISTINCT u.name) as recognized_people
    #                         FROM recognition_sessions s
    #                         LEFT JOIN recognition_logs l ON s.id = l.session_id
    #                         LEFT JOIN users u ON l.user_id = u.id
    #                         WHERE s.id = ?
    #                         GROUP BY s.id
    #                     """, conn, params=(st.session_state.recognition_session_id,))
                        
    #                     if not summary.empty:
    #                         st.success("Recognition Session Completed!")
    #                         st.write("### Session Summary")
    #                         st.write(f"Session Name: {summary['session_name'].iloc[0]}")
    #                         st.write(f"Time: {pd.to_datetime(summary['timestamp'].iloc[0]).strftime('%Y-%m-%d %H:%M:%S')}")
    #                         st.write(f"Total Recognitions: {summary['total_recognitions'].iloc[0]}")
    #                         st.write(f"Unique Faces: {summary['unique_faces'].iloc[0]}")
    #                         st.write(f"Average Confidence: {summary['avg_confidence'].iloc[0]*100:.1f}%")
    #                         st.write(f"Recognized People: {summary['recognized_people'].iloc[0]}")
                
    #             status_placeholder.empty()
    #             video_placeholder.empty()

    # except Exception as e:
    #     logger.error(f"Error in face recognition: {e}")
    #     st.error(f"Error during recognition: {str(e)}")