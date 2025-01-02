import sqlite3
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
import pandas as pd

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
            logging.error(f"Error getting recognition stats: {e}")
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
