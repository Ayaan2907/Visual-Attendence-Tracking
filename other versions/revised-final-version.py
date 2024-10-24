import os
os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'

import cv2
import numpy as np
import sqlite3
import json
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.svm import SVC
import warnings
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, roc_curve, auc
from datetime import datetime
import time
from sklearn.model_selection import cross_val_score
import pickle

warnings.filterwarnings("ignore", category=DeprecationWarning)

class FaceTrainer:
    def __init__(self, confidence_threshold=70, db_path='face_recognition.db'):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.sift = cv2.SIFT_create()
        self.mobilenet_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.db_path = db_path
        self.results_dir = "results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.initialize_database()
        self.migrate_database()
        self.confidence_threshold = confidence_threshold
        self.init_db()

    def initialize_database(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            cursor.execute('''CREATE TABLE IF NOT EXISTS users
                              (id INTEGER PRIMARY KEY AUTOINCREMENT,
                               username TEXT UNIQUE NOT NULL)''')
            
            cursor.execute('''CREATE TABLE IF NOT EXISTS training_sessions
                              (id INTEGER PRIMARY KEY AUTOINCREMENT,
                               timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                               num_users INTEGER,
                               num_images INTEGER,
                               model_accuracy REAL,
                               training_stats TEXT)''')
            
            cursor.execute('''CREATE TABLE IF NOT EXISTS recognition_sessions
                              (id INTEGER PRIMARY KEY AUTOINCREMENT,
                               timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                               total_frames INTEGER,
                               faces_detected INTEGER,
                               faces_recognized INTEGER,
                               recognition_rate REAL,
                               avg_confidence REAL,
                               fps REAL)''')
            
            conn.commit()
            print("Database initialized successfully")
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        finally:
            if conn:
                conn.close()

    def migrate_database(self):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check if the username column exists
            cursor.execute("PRAGMA table_info(training_sessions)")
            columns = [column[1] for column in cursor.fetchall()]
            
            if 'username' not in columns:
                # Add the username column
                cursor.execute("ALTER TABLE training_sessions ADD COLUMN username TEXT")
                conn.commit()
                print("Database schema updated: added 'username' column to training_sessions table")
            
        except sqlite3.Error as e:
            print(f"Database migration error: {e}")
        finally:
            if conn:
                conn.close()

    def preprocess_image(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) == 0:
            return None
        
        (x, y, w, h) = faces[0]
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        
        # Histogram equalization for better contrast
        face = cv2.equalizeHist(face)
        
        return (face, (x, y, w, h))

    def extract_features(self, face):
        # SIFT features
        keypoints, descriptors = self.sift.detectAndCompute(face, None)
        if descriptors is not None:
            sift_features = np.mean(descriptors, axis=0)
        else:
            sift_features = np.zeros(128)  # SIFT descriptor length
        
        # Deep learning features
        face_rgb = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        face_rgb = cv2.resize(face_rgb, (224, 224))
        face_rgb = keras_image.img_to_array(face_rgb)
        face_rgb = np.expand_dims(face_rgb, axis=0)
        face_rgb = tf.keras.applications.mobilenet_v2.preprocess_input(face_rgb)
        deep_features = self.mobilenet_model.predict(face_rgb).flatten()
        
        # Combine all features
        combined_features = np.concatenate([sift_features, deep_features])
        
        return combined_features

    def capture_image_with_retry(self, cap, frame_count, total_imgs):
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera.")
                retry = input("Do you want to retry? (y/n): ").lower()
                if retry == 'y':
                    continue
                else:
                    return None, frame_count

            result = self.preprocess_image(frame)
            if result is not None:
                face, (x, y, w, h) = result
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                frame_count += 1
                img_name = f"image_data/{self.current_username}.{self.current_user_id}.{frame_count}.jpg"
                cv2.imwrite(img_name, face)
                cv2.putText(frame, f"Captured {frame_count}/{total_imgs}", (50, 50), self.font, 0.9, (0, 255, 0), 2)
                print(f"Captured image {frame_count}/{total_imgs}: {img_name}")
                
                cv2.imshow('Capturing Face Data', frame)
                cv2.waitKey(250)  # Display for 250ms
                return frame, frame_count
            else:
                cv2.putText(frame, "No face detected", (50, 50), self.font, 0.9, (0, 0, 255), 2)
                cv2.imshow('Capturing Face Data', frame)
                
                if cv2.waitKey(1) & 0xFF == ord('r'):
                    print("Retrying capture...")
                    continue
                elif cv2.waitKey(1) & 0xFF == ord('s'):
                    print("Skipping this capture...")
                    return None, frame_count

    def generate_dataset(self):
        if not os.path.exists('image_data'):
            os.makedirs('image_data')

        self.current_username = input("Enter the person's name: ")
        
        # Add user to the database if not exists
        self.current_user_id = self.add_user_to_db(self.current_username)
        
        cap = cv2.VideoCapture(0)
        frame_count = 0
        total_imgs = 40

        while frame_count < total_imgs:
            frame, new_frame_count = self.capture_image_with_retry(cap, frame_count, total_imgs)
            if frame is None:
                print(f"Capture failed. Current progress: {frame_count}/{total_imgs}")
                retry = input("Do you want to continue capturing? (y/n): ").lower()
                if retry != 'y':
                    break
            else:
                frame_count = new_frame_count

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"Dataset generation completed for {self.current_username}. {frame_count} images captured.")

        # Update training session in the database
        self.update_training_session(self.current_user_id, frame_count)

    def add_user_to_db(self, username):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("INSERT OR IGNORE INTO users (username) VALUES (?)", (username,))
            conn.commit()
            cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
            user_id = cursor.fetchone()[0]
            return user_id
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        finally:
            if conn:
                conn.close()

    def update_training_session(self, user_id, num_images, accuracy=None, training_stats=None):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO training_sessions 
                (user_id, num_images, model_accuracy, training_stats) 
                VALUES (?, ?, ?, ?)
            """, (user_id, num_images, accuracy, training_stats))
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        finally:
            if conn:
                conn.close()

    def save_model(self, filename='face_recognition_model.pkl'):
        """Save the trained model to a file."""
        if hasattr(self, 'svm_classifier'):
            with open(filename, 'wb') as file:
                pickle.dump(self.svm_classifier, file)
            print(f"Model saved to {filename}")
        else:
            print("No trained model to save. Please train the classifier first.")

    def load_model(self, filename='face_recognition_model.pkl'):
        """Load a trained model from a file."""
        try:
            with open(filename, 'rb') as file:
                loaded_model = pickle.load(file)
                if isinstance(loaded_model, SVC):
                    self.svm_classifier = loaded_model
                    print(f"Model loaded from {filename}")
                    return True
                else:
                    print(f"Error: The loaded object is not an SVC model.")
                    return False
        except FileNotFoundError:
            print(f"Model file {filename} not found.")
        except Exception as e:
            print(f"Error loading model: {e}")
        return False

    def train_classifier(self, retrain=False):
        data_dir = "image_data"
        path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

        faces = []
        ids = []

        print(f"Found {len(path)} images in the data directory.")

        for image_path in path:
            img = cv2.imread(image_path)
            result = self.preprocess_image(img)
            if result is not None:
                face, _ = result
                features = self.extract_features(face)
                id = int(os.path.split(image_path)[1].split('.')[1])
                faces.append(features)
                ids.append(id)
            else:
                print(f"Failed to preprocess image: {image_path}")

        print(f"Successfully processed {len(faces)} images.")

        if len(faces) == 0:
            print("No faces detected in the dataset. Please check your images and try again.")
            return

        faces = np.array(faces)
        ids = np.array(ids)

        unique_ids = np.unique(ids)
        if len(unique_ids) < 2:
            print("Error: At least two different persons are required for training.")
            print(f"Current unique IDs: {unique_ids}")
            return

        X_train, X_test, y_train, y_test = train_test_split(faces, ids, test_size=0.2, random_state=42)

        if retrain and hasattr(self, 'svm_classifier') and isinstance(self.svm_classifier, SVC):
            print("Retraining existing model with new data...")
            self.svm_classifier.fit(X_train, y_train)
        else:
            print("Training new model...")
            self.svm_classifier = SVC(kernel='rbf', probability=True)
            self.svm_classifier.fit(X_train, y_train)

        # After training is complete, save the model
        self.save_model()

        # Performance metrics
        training_start_time = time.time()
        
        # Train SVM classifier
        self.svm_classifier = SVC(kernel='rbf', probability=True)
        self.svm_classifier.fit(X_train, y_train)
        
        training_end_time = time.time()
        training_duration = training_end_time - training_start_time

        # Calculate additional metrics
        y_pred = self.svm_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        
        # Cross-validation score
        cv_scores = cross_val_score(self.svm_classifier, X_train, y_train, cv=5)

        training_stats = {
            "unique_ids": len(unique_ids),
            "total_images": len(faces),
            "images_per_person": len(faces) / len(unique_ids),
            "min_images_per_id": int(min(np.bincount(ids))),  # Convert to int
            "max_images_per_id": int(max(np.bincount(ids))),  # Convert to int
            "training_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            "training_duration": float(training_duration),  # Convert to float
            "accuracy": float(accuracy),  # Convert to float
            "precision": float(precision),  # Convert to float
            "recall": float(recall),  # Convert to float
            "f1_score": float(f1),  # Convert to float
            "cv_mean_score": float(np.mean(cv_scores)),  # Convert to float
            "cv_std": float(np.std(cv_scores)),  # Convert to float
            "confusion_matrix": cm.tolist()  # Convert to list
        }

        # Convert all NumPy types to Python types
        training_stats = {k: self.numpy_to_python(v) for k, v in training_stats.items()}

        # Save to database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO training_sessions 
                (username, num_images, model_accuracy, training_stats) 
                VALUES (?, ?, ?, ?)
            """, ("latest_training", len(faces), float(accuracy), json.dumps(training_stats)))
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        finally:
            if conn:
                conn.close()

        print(f"Training completed. Accuracy: {accuracy:.2f}")
        print(f"Training stats: {training_stats}")

        # Save training stats in visual format
        self.save_training_stats_visual(training_stats, y_test, y_pred, cv_scores)

    def save_training_stats_visual(self, training_stats, y_test, y_pred, cv_scores):
        # Create bar plot for accuracy metrics
        metrics = ['accuracy', 'precision', 'recall', 'f1_score']
        values = [training_stats[metric] for metric in metrics]

        plt.figure(figsize=(10, 6))
        plt.bar(metrics, values)
        plt.title('Model Performance Metrics')
        plt.ylabel('Score')
        plt.ylim(0, 1)
        for i, v in enumerate(values):
            plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
        plt.savefig(os.path.join(self.results_dir, 'model_performance.png'))
        plt.close()

        # Create pie chart for dataset composition
        labels = ['Images per Person', 'Unique IDs']
        sizes = [training_stats['images_per_person'], training_stats['unique_ids']]

        plt.figure(figsize=(8, 8))
        plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
        plt.axis('equal')
        plt.title('Dataset Composition')
        plt.savefig(os.path.join(self.results_dir, 'dataset_composition.png'))
        plt.close()

        # Confusion Matrix Heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(np.array(training_stats['confusion_matrix']), annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix.png'))
        plt.close()

        # ROC Curve (for binary classification)
        if len(np.unique(y_test)) == 2:
            fpr, tpr, _ = roc_curve(y_test, self.svm_classifier.predict_proba(X_test)[:, 1])
            roc_auc = auc(fpr, tpr)
            plt.figure()
            plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate')
            plt.ylabel('True Positive Rate')
            plt.title('Receiver Operating Characteristic (ROC) Curve')
            plt.legend(loc="lower right")
            plt.savefig(os.path.join(self.results_dir, 'roc_curve.png'))
            plt.close()

        # Cross-validation scores distribution
        plt.figure()
        sns.histplot(cv_scores, kde=True)
        plt.title('Distribution of Cross-Validation Scores')
        plt.xlabel('Accuracy')
        plt.ylabel('Frequency')
        plt.savefig(os.path.join(self.results_dir, 'cv_scores_distribution.png'))
        plt.close()

        # Save JSON file
        with open(os.path.join(self.results_dir, 'training_stats.json'), 'w') as f:
            json.dump(training_stats, f, indent=4)

        print(f"Training stats visualizations and JSON saved in {self.results_dir}")

    def init_db(self):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS attendance_data (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                name TEXT,
                timestamp DATETIME,
                detection_score REAL
            )
        ''')
        conn.commit()
        conn.close()

    def save_attendance(self, name, detection_score):
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        cursor.execute('''
            INSERT INTO attendance_data (name, timestamp, detection_score)
            VALUES (?, ?, ?)
        ''', (name, timestamp, detection_score))
        conn.commit()
        conn.close()

    def recognize_face(self):
        cap = cv2.VideoCapture(0)
        recognized_faces = set()  # To keep track of faces already recognized in this session

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                face_img = frame[y:y+h, x:x+w]
                features = self.extract_features(face_img)
                
                if features is not None:
                    features_scaled = self.scaler.transform([features])
                    prediction = self.svm_classifier.predict(features_scaled)
                    proba = self.svm_classifier.predict_proba(features_scaled)[0]
                    confidence = proba.max() * 100
                    
                    if confidence > self.confidence_threshold:
                        label = prediction[0]
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                        cv2.putText(frame, f"ID: {label} ({confidence:.2f}%)", (x, y-10), self.font, 0.9, (0, 255, 0), 2)
                        
                        # Save attendance if not already recognized in this session
                        if label not in recognized_faces:
                            self.save_attendance(label, confidence)
                            recognized_faces.add(label)
                            print(f"Attendance recorded for {label}")
                    else:
                        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                        cv2.putText(frame, "Unknown", (x, y-10), self.font, 0.9, (0, 0, 255), 2)

            cv2.imshow('Face Recognition', frame)
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def numpy_to_python(self, obj):
        # this is mainly to resolve errors from the numpy types saving into json
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return obj

def main():
    trainer = FaceTrainer(confidence_threshold=70, db_path='attendance.db')
    model_filename = 'face_recognition_model.pkl'

    # Check if a trained model is available
    if os.path.exists(model_filename):
        load_choice = input(f"A trained model ({model_filename}) is available. Do you want to load it? (y/n): ").lower()
        if load_choice == 'y':
            if trainer.load_model(model_filename):
                print("Model loaded successfully.")
            else:
                print("Failed to load the model. Proceeding with the main menu.")
        else:
            print("Proceeding without loading the model.")
    else:
        print("No trained model found. You'll need to train a new model.")

    while True:
        print("\nFace Recognition Menu:")
        print("1. Generate Dataset")
        print("2. Train/Retrain Classifier")
        print("3. Recognize Face")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            num_people = int(input("How many people do you want to add to the dataset? "))
            for i in range(num_people):
                trainer.generate_dataset()
            print("Dataset generation completed. Please train the classifier now.")
        elif choice == '2':
            if hasattr(trainer, 'svm_classifier'):
                retrain = input("Existing model found. Do you want to retrain it with new data? (y/n): ").lower() == 'y'
                trainer.train_classifier(retrain=retrain)
            else:
                trainer.train_classifier()
        elif choice == '3':
            if hasattr(trainer, 'svm_classifier'):
                trainer.recognize_face()
            else:
                print("No trained model available. Please train the classifier first.")
        elif choice == '4':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()