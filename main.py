# import os
# os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'

# import cv2
# import numpy as np
# import sqlite3
# import json
# import tensorflow as tf
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.preprocessing import image as keras_image
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
# from sklearn.svm import SVC
# import warnings
# import matplotlib.pyplot as plt
# import seaborn as sns
# from sklearn.metrics import confusion_matrix, roc_curve, auc
# from datetime import datetime
# import time
# from sklearn.model_selection import cross_val_score
# import pickle

# warnings.filterwarnings("ignore", category=DeprecationWarning)

# class FaceTrainer:
#     def __init__(self):
#         self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
#         self.sift = cv2.SIFT_create()
#         self.mobilenet_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
#         self.font = cv2.FONT_HERSHEY_SIMPLEX
#         self.db_path = "face_recognition.db"
#         self.results_dir = "results"
#         if not os.path.exists(self.results_dir):
#             os.makedirs(self.results_dir)
#         self.initialize_database()
#         self.migrate_database()

#     def initialize_database(self):
#         conn = None
#         try:
#             conn = sqlite3.connect(self.db_path)
#             cursor = conn.cursor()
            
#             cursor.execute('''CREATE TABLE IF NOT EXISTS users
#                               (id INTEGER PRIMARY KEY AUTOINCREMENT,
#                                username TEXT UNIQUE NOT NULL)''')
            
#             cursor.execute('''CREATE TABLE IF NOT EXISTS training_sessions
#                               (id INTEGER PRIMARY KEY AUTOINCREMENT,
#                                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
#                                num_users INTEGER,
#                                num_images INTEGER,
#                                model_accuracy REAL,
#                                training_stats TEXT)''')
            
#             cursor.execute('''CREATE TABLE IF NOT EXISTS recognition_sessions
#                               (id INTEGER PRIMARY KEY AUTOINCREMENT,
#                                timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
#                                total_frames INTEGER,
#                                faces_detected INTEGER,
#                                faces_recognized INTEGER,
#                                recognition_rate REAL,
#                                avg_confidence REAL,
#                                fps REAL)''')
#             conn.commit()
#             print("Database initialized successfully")
#         except sqlite3.Error as e:
#             print(f"Database error: {e}")
#         finally:
#             if conn:
#                 conn.close()

#     def migrate_database(self):
#         try:
#             conn = sqlite3.connect(self.db_path)
#             cursor = conn.cursor()
            
#             # Check if the username column exists
#             cursor.execute("PRAGMA table_info(training_sessions)")
#             columns = [column[1] for column in cursor.fetchall()]
            
#             if 'username' not in columns:
#                 # Add the username column
#                 cursor.execute("ALTER TABLE training_sessions ADD COLUMN username TEXT")
#                 conn.commit()
#                 print("Database schema updated: added 'username' column to training_sessions table")
            
#         except sqlite3.Error as e:
#             print(f"Database migration error: {e}")
#         finally:
#             if conn:
#                 conn.close()

#     def preprocess_image(self, image):
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
#         if len(faces) == 0:
#             return None
        
#         (x, y, w, h) = faces[0]
#         face = gray[y:y+h, x:x+w]
#         face = cv2.resize(face, (224, 224))
        
#         # Histogram equalization for better contrast
#         face = cv2.equalizeHist(face)
        
#         return (face, (x, y, w, h))

#     def extract_features(self, face):
#         # SIFT features
#         keypoints, descriptors = self.sift.detectAndCompute(face, None)
#         if descriptors is not None:
#             sift_features = np.mean(descriptors, axis=0)
#         else:
#             sift_features = np.zeros(128)  # SIFT descriptor length
        
#         # Deep learning features
#         face_rgb = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
#         face_rgb = cv2.resize(face_rgb, (224, 224))
#         face_rgb = keras_image.img_to_array(face_rgb)
#         face_rgb = np.expand_dims(face_rgb, axis=0)
#         face_rgb = tf.keras.applications.mobilenet_v2.preprocess_input(face_rgb)
#         deep_features = self.mobilenet_model.predict(face_rgb).flatten()
        
#         # Combine all features
#         combined_features = np.concatenate([sift_features, deep_features])
        
#         return combined_features

#     def capture_image_with_retry(self, cap, frame_count, total_imgs):
#         while True:
#             camera_index = 2  # Change this to the index of the second camera
#             cap = cv2.VideoCapture(camera_index)
#             ret, frame = cap.read()
#             if not ret:
#                 print("Failed to capture frame from camera.")
#                 retry = input("Do you want to retry? (y/n): ").lower()
#                 if retry == 'y':
#                     continue
#                 else:
#                     return None, frame_count

#             result = self.preprocess_image(frame)
#             if result is not None:
#                 face, (x, y, w, h) = result
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
#                 frame_count += 1
#                 img_name = f"image_data/{self.current_username}.{self.current_user_id}.{frame_count}.jpg"
#                 cv2.imwrite(img_name, face)
#                 cv2.putText(frame, f"Captured {frame_count}/{total_imgs}", (50, 50), self.font, 0.9, (0, 255, 0), 2)
#                 print(f"Captured image {frame_count}/{total_imgs}: {img_name}")
                
#                 cv2.imshow('Capturing Face Data', frame)
#                 cv2.waitKey(250)  # Display for 250ms
#                 return frame, frame_count
#             else:
#                 cv2.putText(frame, "No face detected", (50, 50), self.font, 0.9, (0, 0, 255), 2)
#                 cv2.imshow('Capturing Face Data', frame)
                
#                 if cv2.waitKey(1) & 0xFF == ord('r'):
#                     print("Retrying capture...")
#                     continue
#                 elif cv2.waitKey(1) & 0xFF == ord('s'):
#                     print("Skipping this capture...")
#                     return None, frame_count

#     def generate_dataset(self):
#         if not os.path.exists('image_data'):
#             os.makedirs('image_data')

#         self.current_username = input("Enter the person's name: ")
        
#         # Add user to the database if not exists
#         self.current_user_id = self.add_user_to_db(self.current_username)
        
#         cap = cv2.VideoCapture(0)
#         frame_count = 0
#         total_imgs = 20

#         while frame_count < total_imgs:
#             frame, new_frame_count = self.capture_image_with_retry(cap, frame_count, total_imgs)
#             if frame is None:
#                 print(f"Capture failed. Current progress: {frame_count}/{total_imgs}")
#                 retry = input("Do you want to continue capturing? (y/n): ").lower()
#                 if retry != 'y':
#                     break
#             else:
#                 frame_count = new_frame_count

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()
#         print(f"Dataset generation completed for {self.current_username}. {frame_count} images captured.")

#         # Update training session in the database
#         self.update_training_session(self.current_user_id, frame_count)

#     def add_user_to_db(self, username):
#         conn = None
#         try:
#             conn = sqlite3.connect(self.db_path)
#             cursor = conn.cursor()
#             cursor.execute("INSERT OR IGNORE INTO users (username) VALUES (?)", (username,))
#             conn.commit()
#             cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
#             user_id = cursor.fetchone()[0]
#             return user_id
#         except sqlite3.Error as e:
#             print(f"Database error: {e}")
#         finally:
#             if conn:
#                 conn.close()

#     def update_training_session(self, user_id, num_images, accuracy=None, training_stats=None):
#         conn = None
#         try:
#             conn = sqlite3.connect(self.db_path)
#             cursor = conn.cursor()
#             cursor.execute("""
#                 INSERT INTO training_sessions 
#                 (user_id, num_images, model_accuracy, training_stats) 
#                 VALUES (?, ?, ?, ?)
#             """, (user_id, num_images, accuracy, training_stats))
#             conn.commit()
#         except sqlite3.Error as e:
#             print(f"Database error: {e}")
#         finally:
#             if conn:
#                 conn.close()

#     def save_model(self, filename='face_recognition_model.pkl'):
#         """Save the trained model to a file."""
#         if hasattr(self, 'svm_classifier'):
#             with open(filename, 'wb') as file:
#                 pickle.dump(self.svm_classifier, file)
#             print(f"Model saved to {filename}")
#         else:
#             print("No trained model to save. Please train the classifier first.")

#     def load_model(self, filename='face_recognition_model.pkl'):
#         """Load a trained model from a file."""
#         try:
#             with open(filename, 'rb') as file:
#                 self.svm_classifier = pickle.load(file)
#             print(f"Model loaded from {filename}")
#             return True
#         except FileNotFoundError:
#             print(f"Model file {filename} not found.")
#         except Exception as e:
#             print(f"Error loading model: {e}")
#         return False

#     def train_classifier(self, retrain=False):
#         data_dir = "image_data"
#         path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

#         faces = []
#         ids = []

#         print(f"Found {len(path)} images in the data directory.")

#         for image_path in path:
#             img = cv2.imread(image_path)
#             result = self.preprocess_image(img)
#             if result is not None:
#                 face, _ = result
#                 features = self.extract_features(face)
#                 id = int(os.path.split(image_path)[1].split('.')[1])
#                 faces.append(features)
#                 ids.append(id)
#             else:
#                 print(f"Failed to preprocess image: {image_path}")

#         print(f"Successfully processed {len(faces)} images.")

#         if len(faces) == 0:
#             print("No faces detected in the dataset. Please check your images and try again.")
#             return

#         faces = np.array(faces)
#         ids = np.array(ids)

#         unique_ids = np.unique(ids)
#         if len(unique_ids) < 2:
#             print("Error: At least two different persons are required for training.")
#             print(f"Current unique IDs: {unique_ids}")
#             return

#         X_train, X_test, y_train, y_test = train_test_split(faces, ids, test_size=0.2, random_state=42)

#         if retrain and hasattr(self, 'svm_classifier'):
#             print("Retraining existing model with new data...")
#             # Assuming X_train and y_train are your new training data
#             self.svm_classifier.fit(X_train, y_train)
#         else:
#             print("Training new model...")
#             self.svm_classifier = SVC(kernel='rbf', probability=True)
#             self.svm_classifier.fit(X_train, y_train)

#         # After training is complete, save the model
#         self.save_model()

#         # Performance metrics
#         training_start_time = time.time()
        
#         # Train SVM classifier
#         self.svm_classifier = SVC(kernel='rbf', probability=True)
#         self.svm_classifier.fit(X_train, y_train)
        
#         training_end_time = time.time()
#         training_duration = training_end_time - training_start_time

#         # Calculate additional metrics
#         y_pred = self.svm_classifier.predict(X_test)
#         accuracy = accuracy_score(y_test, y_pred)
#         precision = precision_score(y_test, y_pred, average='weighted')
#         recall = recall_score(y_test, y_pred, average='weighted')
#         f1 = f1_score(y_test, y_pred, average='weighted')
        
#         # Confusion matrix
#         cm = confusion_matrix(y_test, y_pred)
        
#         # Cross-validation score
#         cv_scores = cross_val_score(self.svm_classifier, X_train, y_train, cv=5)

#         training_stats = {
#             "unique_ids": len(unique_ids),
#             "total_images": len(faces),
#             "images_per_person": len(faces) / len(unique_ids),
#             "min_images_per_id": int(min(np.bincount(ids))),  # Convert to int
#             "max_images_per_id": int(max(np.bincount(ids))),  # Convert to int
#             "training_date": datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
#             "training_duration": float(training_duration),  # Convert to float
#             "accuracy": float(accuracy),  # Convert to float
#             "precision": float(precision),  # Convert to float
#             "recall": float(recall),  # Convert to float
#             "f1_score": float(f1),  # Convert to float
#             "cv_mean_score": float(np.mean(cv_scores)),  # Convert to float
#             "cv_std": float(np.std(cv_scores)),  # Convert to float
#             "confusion_matrix": cm.tolist()  # Convert to list
#         }

#         # Convert all NumPy types to Python types
#         training_stats = {k: self.numpy_to_python(v) for k, v in training_stats.items()}

#         # Save to database
#         try:
#             conn = sqlite3.connect(self.db_path)
#             cursor = conn.cursor()
#             cursor.execute("""
#                 INSERT INTO training_sessions 
#                 (username, num_images, model_accuracy, training_stats) 
#                 VALUES (?, ?, ?, ?)
#             """, ("latest_training", len(faces), float(accuracy), json.dumps(training_stats)))
#             conn.commit()
#         except sqlite3.Error as e:
#             print(f"Database error: {e}")
#         finally:
#             if conn:
#                 conn.close()

#         print(f"Training completed. Accuracy: {accuracy:.2f}")
#         print(f"Training stats: {training_stats}")

#         # Save training stats in visual format
#         self.save_training_stats_visual(training_stats, y_test, y_pred, cv_scores, X_test)  # Pass X_test

#     def save_training_stats_visual(self, training_stats, y_test, y_pred, cv_scores, X_test):
#         # Create bar plot for accuracy metrics
#         metrics = ['accuracy', 'precision', 'recall', 'f1_score']
#         values = [training_stats[metric] for metric in metrics]

#         plt.figure(figsize=(10, 6))
#         plt.bar(metrics, values)
#         plt.title('Model Performance Metrics')
#         plt.ylabel('Score')
#         plt.ylim(0, 1)
#         for i, v in enumerate(values):
#             plt.text(i, v, f'{v:.2f}', ha='center', va='bottom')
#         plt.savefig(os.path.join(self.results_dir, 'model_performance.png'))
#         plt.close()

#         # Create pie chart for dataset composition
#         labels = ['Images per Person', 'Unique IDs']
#         sizes = [training_stats['images_per_person'], training_stats['unique_ids']]

#         plt.figure(figsize=(8, 8))
#         plt.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=90)
#         plt.axis('equal')
#         plt.title('Dataset Composition')
#         plt.savefig(os.path.join(self.results_dir, 'dataset_composition.png'))
#         plt.close()

#         # Confusion Matrix Heatmap
#         plt.figure(figsize=(10, 8))
#         sns.heatmap(np.array(training_stats['confusion_matrix']), annot=True, fmt='d', cmap='Blues')
#         plt.title('Confusion Matrix')
#         plt.ylabel('True Label')
#         plt.xlabel('Predicted Label')
#         plt.savefig(os.path.join(self.results_dir, 'confusion_matrix.png'))
#         plt.close()

#         # ROC Curve (for binary classification)
#         if len(np.unique(y_test)) == 2:
#             # Assuming 1 is the positive class and 3 is the negative class
#             y_test_binary = np.where(y_test == 1, 1, 0)
#             y_pred_binary = np.where(y_pred == 1, 1, 0)

#             # Then use y_test_binary and y_pred_binary for roc_curve
#             fpr, tpr, _ = roc_curve(y_test_binary, self.svm_classifier.predict_proba(X_test)[:, 1])
#             roc_auc = auc(fpr, tpr)
#             plt.figure()
#             plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
#             plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
#             plt.xlim([0.0, 1.0])
#             plt.ylim([0.0, 1.05])
#             plt.xlabel('False Positive Rate')
#             plt.ylabel('True Positive Rate')
#             plt.title('Receiver Operating Characteristic (ROC) Curve')
#             plt.legend(loc="lower right")
#             plt.savefig(os.path.join(self.results_dir, 'roc_curve.png'))
#             plt.close()

#         # Cross-validation scores distribution
#         plt.figure()
#         sns.histplot(cv_scores, kde=True)
#         plt.title('Distribution of Cross-Validation Scores')
#         plt.xlabel('Accuracy')
#         plt.ylabel('Frequency')
#         plt.savefig(os.path.join(self.results_dir, 'cv_scores_distribution.png'))
#         plt.close()

#         # Save JSON file
#         with open(os.path.join(self.results_dir, 'training_stats.json'), 'w') as f:
#             json.dump(training_stats, f, indent=4)

#         print(f"Training stats visualizations and JSON saved in {self.results_dir}")

#     def recognize_face(self):
#         cap = cv2.VideoCapture(0)
#         recognition_stats = {
#             "total_frames": 0,
#             "faces_detected": 0,
#             "faces_recognized": 0,
#             "confidence_scores": []
#         }
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("Failed to capture frame from camera.")
#                 break

#             result = self.preprocess_image(frame)
#             if result is not None:
#                 face, (x, y, w, h) = result
#                 features = self.extract_features(face)
                
#                 # Predict using SVM classifier
#                 prediction = self.svm_classifier.predict([features])
#                 confidence = self.svm_classifier.predict_proba([features]).max() * 100

#                 recognition_stats["faces_detected"] += 1
#                 recognition_stats["confidence_scores"].append(confidence)

#                 if confidence > 50:
#                     recognition_stats["faces_recognized"] += 1
#                     label = prediction[0]
#                     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
#                     cv2.putText(frame, f"ID: {label} ({confidence:.2f}%)", (x, y-10), self.font, 0.9, (0, 255, 0), 2)
#                 else:
#                     cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
#                     cv2.putText(frame, "Unknown", (x, y-10), self.font, 0.9, (0, 0, 255), 2)
#             else:
#                 cv2.putText(frame, "No face detected", (50, 50), self.font, 0.9, (0, 0, 255), 2)

#             cv2.imshow('Face Recognition', frame)
#             recognition_stats["total_frames"] += 1

#             if cv2.waitKey(1) & 0xFF == ord('q'):
#                 break

#         cap.release()
#         cv2.destroyAllWindows()

#         recognition_rate = recognition_stats["faces_recognized"] / recognition_stats["faces_detected"] if recognition_stats["faces_detected"] > 0 else 0
#         avg_confidence = np.mean(recognition_stats["confidence_scores"]) if recognition_stats["confidence_scores"] else 0
        
#         print(f"Recognition stats: {recognition_stats}")
#         print(f"Recognition rate: {recognition_rate:.2f}")
#         print(f"Average confidence: {avg_confidence:.2f}")

#         # Save recognition stats to database
#         try:
#             conn = sqlite3.connect(self.db_path)
#             cursor = conn.cursor()
#             cursor.execute("""
#                 INSERT INTO recognition_sessions 
#                 (total_frames, faces_detected, faces_recognized, recognition_rate, avg_confidence) 
#                 VALUES (?, ?, ?, ?, ?)
#             """, (recognition_stats["total_frames"], recognition_stats["faces_detected"], 
#                   recognition_stats["faces_recognized"], recognition_rate, avg_confidence))
#             conn.commit()
#         except sqlite3.Error as e:
#             print(f"Database error: {e}")
#         finally:
#             if conn:
#                 conn.close()

#     def numpy_to_python(self, obj):
#         # this is mainly to resolve errors from the numpy types saving into json
#         if isinstance(obj, np.integer):
#             return int(obj)
#         elif isinstance(obj, np.floating):
#             return float(obj)
#         elif isinstance(obj, np.ndarray):
#             return obj.tolist()
#         else:
#             return obj

# def main():
#     trainer = FaceTrainer()
#     model_filename = 'face_recognition_model.pkl'
#     while True:
#         print("\nFace Recognition Menu:")
#         print("1. Generate Dataset")
#         print("2. Train/Retrain Classifier")
#         print("3. Recognize Face")
#         print("4. Exit")
        
#         choice = input("Enter your choice (1-4): ")   
#         if choice == '1':
#             trainer.generate_dataset()
#         elif choice == '2':
#             trainer.train_classifier(retrain=retrain)
#         elif choice == '3':
#             if hasattr(trainer, 'svm_classifier'):
#                 trainer.recognize_face()
#             else:
#                 print("No trained model available. Please train the classifier first.")
#         elif choice == '4':
#             print("Exiting the program.")
#             break
#         else:
#             print("Invalid choice. Please try again.")

# if __name__ == "__main__":
#     main()


# import os
# os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'

# import streamlit as st
# import cv2
# import numpy as np
# import sqlite3
# import json
# import tensorflow as tf
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.preprocessing import image as keras_image
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
#                               f1_score, confusion_matrix, roc_curve, auc)
# from sklearn.svm import SVC
# import warnings
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime
# import time
# from sklearn.model_selection import cross_val_score
# import pickle
# import base64
# import pandas as pd

# # Suppress warnings
# warnings.filterwarnings("ignore")

# class FaceRecognitionSystem:
#     def __init__(self):
#         # Initialize core components
#         self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         self.sift = cv2.SIFT_create()
#         self.mobilenet_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        
#         # Paths and directories
#         self.db_path = "face_recognition.db"
#         self.results_dir = "results"
#         self.image_data_dir = "image_data"
        
#         # Create necessary directories
#         for dir_path in [self.results_dir, self.image_data_dir]:
#             os.makedirs(dir_path, exist_ok=True)
        
#         # Initialize database
#         self._initialize_database()

#     def _initialize_database(self):
#         """Initialize SQLite database with required tables"""
#         try:
#             with sqlite3.connect(self.db_path) as conn:
#                 cursor = conn.cursor()
                
#                 # Users table
#                 cursor.execute('''CREATE TABLE IF NOT EXISTS users
#                                   (id INTEGER PRIMARY KEY AUTOINCREMENT,
#                                    username TEXT UNIQUE NOT NULL,
#                                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP)''')
                
#                 # Training sessions table
#                 cursor.execute('''CREATE TABLE IF NOT EXISTS training_sessions
#                                   (id INTEGER PRIMARY KEY AUTOINCREMENT,
#                                    username TEXT,
#                                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
#                                    num_images INTEGER,
#                                    model_accuracy REAL,
#                                    training_stats TEXT)''')
                
#                 # Recognition sessions table
#                 cursor.execute('''CREATE TABLE IF NOT EXISTS recognition_sessions
#                                   (id INTEGER PRIMARY KEY AUTOINCREMENT,
#                                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
#                                    total_frames INTEGER,
#                                    faces_detected INTEGER,
#                                    faces_recognized INTEGER,
#                                    recognition_rate REAL,
#                                    avg_confidence REAL)''')
                
#                 conn.commit()
#         except sqlite3.Error as e:
#             st.error(f"Database initialization error: {e}")

#     def preprocess_image(self, image):
#         """Preprocess image for face detection and feature extraction"""
#         gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#         faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
#         if len(faces) == 0:
#             return None
        
#         (x, y, w, h) = faces[0]
#         face = gray[y:y+h, x:x+w]
#         face = cv2.resize(face, (224, 224))
#         face = cv2.equalizeHist(face)
        
#         return (face, (x, y, w, h))

#     def extract_features(self, face):
#         """Extract combined features using SIFT and MobileNetV2"""
#         # SIFT features
#         keypoints, descriptors = self.sift.detectAndCompute(face, None)
#         sift_features = np.mean(descriptors, axis=0) if descriptors is not None else np.zeros(128)
        
#         # Deep learning features
#         face_rgb = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
#         face_rgb = cv2.resize(face_rgb, (224, 224))
#         face_rgb = keras_image.img_to_array(face_rgb)
#         face_rgb = np.expand_dims(face_rgb, axis=0)
#         face_rgb = tf.keras.applications.mobilenet_v2.preprocess_input(face_rgb)
#         deep_features = self.mobilenet_model.predict(face_rgb).flatten()
        
#         # Combine features
#         return np.concatenate([sift_features, deep_features])

#     def generate_dataset(self, username, num_images=20):
#         """Capture training images for a user"""
#         user_id = self._add_user(username)
#         image_paths = []
        
#         cap = cv2.VideoCapture(0)
#         frame_count = 0
        
#         while frame_count < num_images:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             result = self.preprocess_image(frame)
#             if result is not None:
#                 face, (x, y, w, h) = result
#                 frame_count += 1
                
#                 # Save face image
#                 img_path = os.path.join(self.image_data_dir, f"{username}.{user_id}.{frame_count}.jpg")
#                 cv2.imwrite(img_path, face)
#                 image_paths.append(img_path)
        
#         cap.release()
        
#         return image_paths

#     def _add_user(self, username):
#         """Add user to database"""
#         try:
#             with sqlite3.connect(self.db_path) as conn:
#                 cursor = conn.cursor()
#                 cursor.execute("INSERT OR IGNORE INTO users (username) VALUES (?)", (username,))
#                 conn.commit()
#                 cursor.execute("SELECT id FROM users WHERE username = ?", (username,))
#                 return cursor.fetchone()[0]
#         except sqlite3.Error as e:
#             st.error(f"Error adding user: {e}")
#             return None

#     def train_classifier(self, retrain=False):
#         """Train SVM classifier for face recognition"""
#         # Load images and extract features
#         image_paths = [os.path.join(self.image_data_dir, f) for f in os.listdir(self.image_data_dir) if f.endswith('.jpg')]
        
#         faces, labels = [], []
#         for path in image_paths:
#             img = cv2.imread(path)
#             result = self.preprocess_image(img)
#             if result is not None:
#                 face, _ = result
#                 features = self.extract_features(face)
#                 user_id = int(os.path.splitext(os.path.basename(path))[0].split('.')[1])
#                 faces.append(features)
#                 labels.append(user_id)
        
#         # Prepare data for training
#         X = np.array(faces)
#         y = np.array(labels)
        
#         # Split data
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
#         # Train classifier
#         self.svm_classifier = SVC(kernel='rbf', probability=True)
#         self.svm_classifier.fit(X_train, y_train)
        
#         # Evaluate model
#         y_pred = self.svm_classifier.predict(X_test)
#         metrics = {
#             'accuracy': accuracy_score(y_test, y_pred),
#             'precision': precision_score(y_test, y_pred, average='weighted'),
#             'recall': recall_score(y_test, y_pred, average='weighted'),
#             'f1_score': f1_score(y_test, y_pred, average='weighted')
#         }
        
#         # Save model
#         with open('face_recognition_model.pkl', 'wb') as f:
#             pickle.dump(self.svm_classifier, f)
        
#         return metrics

#     def recognize_faces(self):
#         """Real-time face recognition"""
#         cap = cv2.VideoCapture(0)
#         recognized_faces = []
        
#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 break
            
#             result = self.preprocess_image(frame)
#             if result is not None:
#                 face, (x, y, w, h) = result
#                 features = self.extract_features(face)
                
#                 # Predict
#                 prediction = self.svm_classifier.predict([features])
#                 confidence = self.svm_classifier.predict_proba([features]).max() * 100
                
#                 if confidence > 50:
#                     recognized_faces.append({
#                         'user_id': prediction[0],
#                         'confidence': confidence
#                     })
                
#                 # Draw rectangle and label
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), 
#                               (0, 255, 0) if confidence > 50 else (0, 0, 255), 2)
#                 label = f"User {prediction[0]} ({confidence:.2f}%)" if confidence > 50 else "Unknown"
#                 cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, 
#                             (0, 255, 0) if confidence > 50 else (0, 0, 255), 2)
            
#             yield frame, recognized_faces

# def main():
#     st.set_page_config(page_title="Advanced Face Recognition", layout="wide")
    
#     # Initialize session state
#     if 'system' not in st.session_state:
#         st.session_state.system = FaceRecognitionSystem()
    
#     # Sidebar navigation
#     st.sidebar.title("Face Recognition System")
#     menu_options = [
#         "Dashboard", 
#         "Dataset Generation", 
#         "Model Training", 
#         "Face Recognition", 
#         "Database Viewer",
#         "About"
#     ]
#     selected_menu = st.sidebar.radio("Navigation", menu_options)
    
#     # Main content area
#     if selected_menu == "Dashboard":
#         st.title("Face Recognition Dashboard")
        
#         # Stats from database
#         with sqlite3.connect(st.session_state.system.db_path) as conn:
#             users_count = pd.read_sql("SELECT COUNT(*) as count FROM users", conn).iloc[0]['count']
#             training_sessions = pd.read_sql("SELECT * FROM training_sessions ORDER BY timestamp DESC LIMIT 5", conn)
#             recognition_sessions = pd.read_sql("SELECT * FROM recognition_sessions ORDER BY timestamp DESC LIMIT 5", conn)
        
#         col1, col2, col3 = st.columns(3)
#         with col1:
#             st.metric("Total Users", users_count)
#         with col2:
#             st.metric("Last Training", training_sessions['timestamp'].iloc[0] if not training_sessions.empty else "N/A")
#         with col3:
#             st.metric("Last Recognition", recognition_sessions['timestamp'].iloc[0] if not recognition_sessions.empty else "N/A")
        
#         # Recent training sessions
#         st.subheader("Recent Training Sessions")
#         st.dataframe(training_sessions)
    
#     elif selected_menu == "Dataset Generation":
#         st.title("Generate Face Dataset")
        
#         username = st.text_input("Enter Username", key="dataset_username")
#         num_images = st.slider("Number of Training Images", 10, 50, 20)
        
#         if st.button("Start Capturing"):
#             if username:
#                 with st.spinner("Capturing images..."):
#                     image_paths = st.session_state.system.generate_dataset(username, num_images)
                
#                 st.success(f"Captured {len(image_paths)} images for {username}")
                
#                 # Display captured images in a grid
#                 cols = st.columns(5)
#                 for i, path in enumerate(image_paths[:5]):
#                     with cols[i]:
#                         st.image(path)
#             else:
#                 st.warning("Please enter a username")
    
#     elif selected_menu == "Model Training":
#         st.title("Train Face Recognition Model")
        
#         if st.button("Train Model"):
#             with st.spinner("Training model..."):
#                 metrics = st.session_state.system.train_classifier()
            
#             # Display metrics
#             col1, col2, col3, col4 = st.columns(4)
#             with col1:
#                 st.metric("Accuracy", f"{metrics['accuracy']:.2%}")
#             with col2:
#                 st.metric("Precision", f"{metrics['precision']:.2%}")
#             with col3:
#                 st.metric("Recall", f"{metrics['recall']:.2%}")
#             with col4:
#                 st.metric("F1 Score", f"{metrics['f1_score']:.2%}")
    
#     elif selected_menu == "Face Recognition":
#         st.title("Real-time Face Recognition")
        
#         # Check if model is trained
#         model_path = 'face_recognition_model.pkl'
#         if not os.path.exists(model_path):
#             st.warning("Please train the model first")
#         else:
#             # Load trained model
#             with open(model_path, 'rb') as f:
#                 st.session_state.system.svm_classifier = pickle.load(f)
            
#             # Video stream
#             stframe = st.empty()
#             recognition_stats = st.empty()
            
#             recognized_users = []
#             for frame, faces in st.session_state.system.recognize_faces():
#                 # Convert frame to RGB for Streamlit
#                 frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#                 stframe.image(frame_rgb)
                
#                 # Update recognized faces
#                 for face in faces:
#                     if face not in recognized_users:
#                         recognized_users.append(face)
                
#                 # Display recognized users
#                 recognition_stats.write(f"Recognized Users: {len(recognized_users)}")
    
#     elif selected_menu == "Database Viewer":
#         st.title("Database Information")
        
#         # Tabs for different views
#         users_tab, training_tab, recognition_tab = st.tabs(["Users", "Training Sessions", "Recognition Sessions"])
        
#         with sqlite3.connect(st.session_state.system.db_path) as conn:
#             with users_tab:
#                 users_df = pd.read_sql("SELECT * FROM users", conn)
#                 st.dataframe(users_df)
            
#             with training_tab:
#                 training_df = pd.read_sql("SELECT * FROM training_sessions", conn)
#                 st.dataframe(training_df)
            
#             with recognition_tab:
#                 recognition_df = pd.read_sql("SELECT * FROM recognition_sessions", conn)
#                 st.dataframe(recognition_df)
    
#     elif selected_menu == "About":
#         st.title("About Face Recognition System")
#         st.markdown("""
#         ### Advanced Face Recognition System
        
#         **Features:**
#         - Dataset generation for multiple users
#         - Machine Learning-based face recognition
#         - Real-time face detection and identification
#         - Comprehensive database tracking
        
#         **Technologies Used:**
#         - OpenCV for image processing
#         - TensorFlow for deep learning features
#         - scikit-learn for machine learning
#         - Streamlit for interactive web application
#         """)

# if __name__ == "__main__":
#     main()




# import os
# os.environ['OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS'] = '0'

# import streamlit as st
# import cv2
# import numpy as np
# import sqlite3
# import json
# import tensorflow as tf
# from tensorflow.keras.applications import MobileNetV2
# from tensorflow.keras.preprocessing import image as keras_image
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import (accuracy_score, precision_score, recall_score, 
#                            f1_score, confusion_matrix, roc_curve, auc)
# from sklearn.svm import SVC
# import warnings
# import matplotlib.pyplot as plt
# import seaborn as sns
# from datetime import datetime
# import time
# from sklearn.model_selection import cross_val_score
# import pickle
# import pandas as pd
# from pathlib import Path
# import logging
# import sys


# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Suppress tensorflow warnings
# tf.get_logger().setLevel('ERROR')

# class FaceRecognitionSystem:
#     def __init__(self):
#         self.setup_directories()
#         self.initialize_models()
#         self.initialize_database()
#         self.load_classifier()

#     def setup_directories(self):
#         """Set up necessary directories"""
#         self.base_dir = Path('face_recognition_data')
#         self.db_path = self.base_dir / 'face_recognition.db'
#         self.model_dir = self.base_dir / 'models'
#         self.image_dir = self.base_dir / 'images'
        
#         for directory in [self.base_dir, self.model_dir, self.image_dir]:
#             directory.mkdir(parents=True, exist_ok=True)

#     def initialize_models(self):
#         """Initialize CV and ML models"""
#         self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         self.mobilenet = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')

#     def initialize_database(self):
#         """Initialize SQLite database"""
#         try:
#             with sqlite3.connect(self.db_path) as conn:
#                 cursor = conn.cursor()
                
#                 # Users table
#                 cursor.execute('''
#                     CREATE TABLE IF NOT EXISTS users (
#                         id INTEGER PRIMARY KEY AUTOINCREMENT,
#                         name TEXT UNIQUE NOT NULL,
#                         created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#                         image_count INTEGER DEFAULT 0
#                     )
#                 ''')
                
#                 # Training sessions table
#                 cursor.execute('''
#                     CREATE TABLE IF NOT EXISTS training_sessions (
#                         id INTEGER PRIMARY KEY AUTOINCREMENT,
#                         timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
#                         num_users INTEGER,
#                         accuracy REAL,
#                         metrics TEXT
#                     )
#                 ''')
                
#                 conn.commit()
#                 logger.info("Database initialized successfully")
#         except Exception as e:
#             logger.error(f"Database initialization error: {e}")
#             raise

#     def load_classifier(self):
#         """Load trained classifier if exists"""
#         model_path = self.model_dir / 'face_classifier.pkl'
#         if model_path.exists():
#             with open(model_path, 'rb') as f:
#                 self.classifier = pickle.load(f)
#                 return True
#         return False

#     def save_classifier(self):
#         """Save trained classifier"""
#         model_path = self.model_dir / 'face_classifier.pkl'
#         with open(model_path, 'wb') as f:
#             pickle.dump(self.classifier, f)

#     def detect_face(self, frame):
#         """Detect face in frame and return preprocessed face image"""
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
#         if len(faces) == 0:
#             return None, None
        
#         (x, y, w, h) = faces[0]  # Take the first face
#         face = frame[y:y+h, x:x+w]
#         face = cv2.resize(face, (224, 224))
        
#         return face, (x, y, w, h)

#     def extract_features(self, face):
#         """Extract features using MobileNetV2"""
#         face = cv2.resize(face, (224, 224))
#         face = keras_image.img_to_array(face)
#         face = np.expand_dims(face, axis=0)
#         face = tf.keras.applications.mobilenet_v2.preprocess_input(face)
#         features = self.mobilenet.predict(face, verbose=0)
#         return features.flatten()

#     def capture_dataset(self, username, num_images=20):
#         """Capture dataset for a user"""
#         user_id = self.add_user(username)
#         if user_id is None:
#             raise ValueError(f"Could not create user: {username}")

#         image_paths = []
#         cap = cv2.VideoCapture(0)
        
#         # Create progress bar
#         progress_text = "Capturing images..."
#         progress_bar = st.progress(0)
#         status_text = st.empty()

#         for i in range(num_images):
#             ret, frame = cap.read()
#             if not ret:
#                 break

#             face, bbox = self.detect_face(frame)
#             if face is not None:
#                 # Save face image
#                 image_path = self.image_dir / f"{username}_{user_id}_{i}.jpg"
#                 cv2.imwrite(str(image_path), face)
#                 image_paths.append(image_path)

#                 # Update progress
#                 progress = (i + 1) / num_images
#                 progress_bar.progress(progress)
#                 status_text.text(f"Captured {i+1}/{num_images} images")
#                 time.sleep(0.1)  # Small delay between captures

#         cap.release()
        
#         # Update user's image count
#         self.update_user_image_count(user_id, len(image_paths))
#         return image_paths

#     def train_model(self):
#         """Train face recognition model"""
#         features = []
#         labels = []
#         user_counts = {}

#         # Load and process all images
#         for img_path in self.image_dir.glob("*.jpg"):
#             username = img_path.stem.split('_')[0]
#             user_id = int(img_path.stem.split('_')[1])
            
#             image = cv2.imread(str(img_path))
#             face_features = self.extract_features(image)
            
#             features.append(face_features)
#             labels.append(user_id)
#             user_counts[user_id] = user_counts.get(user_id, 0) + 1

#         if len(set(labels)) < 2:
#             raise ValueError("Need at least 2 different users for training")

#         # Convert to numpy arrays
#         X = np.array(features)
#         y = np.array(labels)

#         # Split dataset
#         X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#         # Train classifier
#         self.classifier = SVC(kernel='rbf', probability=True)
#         self.classifier.fit(X_train, y_train)

#         # Evaluate
#         y_pred = self.classifier.predict(X_test)
#         metrics = {
#             'accuracy': float(accuracy_score(y_test, y_pred)),
#             'precision': float(precision_score(y_test, y_pred, average='weighted')),
#             'recall': float(recall_score(y_test, y_pred, average='weighted')),
#             'f1_score': float(f1_score(y_test, y_pred, average='weighted'))
#         }

#         # Save model and metrics
#         self.save_classifier()
#         self.save_training_session(metrics, len(user_counts))

#         return metrics

#     def recognize_face(self, frame):
#         """Recognize face in frame"""
#         face, bbox = self.detect_face(frame)
#         if face is None:
#             return None, None, None

#         features = self.extract_features(face)
#         prediction = self.classifier.predict([features])[0]
#         confidence = float(self.classifier.predict_proba([features]).max())

#         return prediction, confidence, bbox

#     def add_user(self, username):
#         """Add new user to database"""
#         try:
#             with sqlite3.connect(self.db_path) as conn:
#                 cursor = conn.cursor()
#                 cursor.execute("INSERT OR IGNORE INTO users (name) VALUES (?)", (username,))
#                 cursor.execute("SELECT id FROM users WHERE name = ?", (username,))
#                 return cursor.fetchone()[0]
#         except Exception as e:
#             logger.error(f"Error adding user: {e}")
#             return None

#     def update_user_image_count(self, user_id, count):
#         """Update user's image count"""
#         try:
#             with sqlite3.connect(self.db_path) as conn:
#                 cursor = conn.cursor()
#                 cursor.execute("UPDATE users SET image_count = ? WHERE id = ?", (count, user_id))
#         except Exception as e:
#             logger.error(f"Error updating image count: {e}")

#     def save_training_session(self, metrics, num_users):
#         """Save training session metrics"""
#         try:
#             with sqlite3.connect(self.db_path) as conn:
#                 cursor = conn.cursor()
#                 cursor.execute(
#                     "INSERT INTO training_sessions (num_users, accuracy, metrics) VALUES (?, ?, ?)",
#                     (num_users, metrics['accuracy'], json.dumps(metrics))
#                 )
#         except Exception as e:
#             logger.error(f"Error saving training session: {e}")

# def main():
#     st.set_page_config(page_title="Face Recognition System", layout="wide")

#     # Initialize system
#     if 'system' not in st.session_state:
#         st.session_state.system = FaceRecognitionSystem()

#     # Sidebar navigation
#     st.sidebar.title("Face Recognition System")
#     page = st.sidebar.radio("Navigation", [
#         "Dashboard",
#         "Dataset Collection",
#         "Model Training",
#         "Face Recognition",
#         "System Status"
#     ])

#     if page == "Dashboard":
#         show_dashboard()
#     elif page == "Dataset Collection":
#         show_dataset_collection()
#     elif page == "Model Training":
#         show_model_training()
#     elif page == "Face Recognition":
#         show_face_recognition()
#     elif page == "System Status":
#         show_system_status()

# def show_dashboard():
#     st.title("Face Recognition Dashboard")

#     # System stats
#     with sqlite3.connect(st.session_state.system.db_path) as conn:
#         users_df = pd.read_sql("SELECT COUNT(*) as count FROM users", conn)
#         training_df = pd.read_sql(
#             "SELECT accuracy FROM training_sessions ORDER BY timestamp DESC LIMIT 1", 
#             conn
#         )

#     col1, col2 = st.columns(2)
#     with col1:
#         st.metric("Registered Users", users_df['count'].iloc[0])
#     with col2:
#         if not training_df.empty:
#             st.metric("Latest Model Accuracy", f"{training_df['accuracy'].iloc[0]:.2%}")
#         else:
#             st.metric("Latest Model Accuracy", "No model trained")

# def show_dataset_collection():
#     st.title("Dataset Collection")

#     # Camera preview
#     st.subheader("Camera Preview")
#     preview = st.empty()
    
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         st.error("Error: Could not access camera")
#         return

#     # Form for dataset collection
#     with st.form("dataset_form"):
#         username = st.text_input("Enter username")
#         num_images = st.slider("Number of images to capture", 10, 50, 20)
#         submit = st.form_submit_button("Start Capture")

#     if submit and username:
#         try:
#             image_paths = st.session_state.system.capture_dataset(username, num_images)
#             st.success(f"Successfully captured {len(image_paths)} images")

#             # Display sample images
#             st.subheader("Sample Images")
#             cols = st.columns(5)
#             for idx, img_path in enumerate(image_paths[:5]):
#                 cols[idx].image(str(img_path))
#         except Exception as e:
#             st.error(f"Error during capture: {str(e)}")

#     # Show live preview
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             break
        
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         preview.image(frame_rgb, channels="RGB")

#         if st.button("Stop Preview"):
#             break

#     cap.release()

# def show_model_training():
#     st.title("Model Training")

#     # Display current users
#     with sqlite3.connect(st.session_state.system.db_path) as conn:
#         users_df = pd.read_sql("SELECT name, image_count FROM users", conn)
    
#     st.subheader("Current Users")
#     st.dataframe(users_df)

#     if st.button("Train Model"):
#         try:
#             with st.spinner("Training model..."):
#                 metrics = st.session_state.system.train_model()
            
#             # Display metrics
#             col1, col2, col3, col4 = st.columns(4)
#             col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
#             col2.metric("Precision", f"{metrics['precision']:.2%}")
#             col3.metric("Recall", f"{metrics['recall']:.2%}")
#             col4.metric("F1 Score", f"{metrics['f1_score']:.2%}")
            
#             st.success("Model training completed!")
#         except ValueError as e:
#             st.error(str(e))
#         except Exception as e:
#             st.error(f"Error during training: {str(e)}")

# def show_face_recognition():
#     st.title("Face Recognition")

#     if not hasattr(st.session_state.system, 'classifier'):
#         st.warning("Please train the model first")
#         return

#     # Initialize video capture
#     cap = cv2.VideoCapture(0)
#     if not cap.isOpened():
#         st.error("Error: Could not access camera")
#         return

#     # Video frame placeholder
#     frame_placeholder = st.empty()
#     stop_button = st.button("Stop Recognition")

#     while not stop_button:
#         ret, frame = cap.read()
#         if not ret:
#             break

#         # Recognize face
#         prediction, confidence, bbox = st.session_state.system.recognize_face(frame)
        
#         if bbox is not None:
#             x, y, w, h = bbox
#             color = (0, 255, 0) if confidence > 0.5 else (0, 0, 255)
#             cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            
#             if prediction is not None and confidence > 0.5:
#                 label = f"User {prediction} ({confidence:.2%})"
#                 cv2.putText(frame, label, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

#         # Display frame
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame_placeholder.image(frame_rgb, channels="RGB")

#     cap.release()

# # Complete the previous code by adding:

# def show_system_status():
#     st.title("System Status")
    
#     # System Information
#     st.subheader("System Information")
#     col1, col2 = st.columns(2)
    
#     with col1:
#         st.info("Storage Information")
#         total_size = 0
#         num_images = 0
        
#         # Calculate storage usage
#         for filepath in st.session_state.system.image_dir.glob("**/*"):
#             if filepath.is_file():
#                 total_size += filepath.stat().st_size
#                 if filepath.suffix.lower() in ['.jpg', '.jpeg', '.png']:
#                     num_images += 1
        
#         st.write(f"Total Images: {num_images}")
#         st.write(f"Storage Used: {total_size / (1024*1024):.2f} MB")
        
#         # Model status
#         model_exists = hasattr(st.session_state.system, 'classifier')
#         st.write(f"Model Status: {'Trained' if model_exists else 'Not Trained'}")
    
#     with col2:
#         st.info("Database Information")
#         try:
#             with sqlite3.connect(st.session_state.system.db_path) as conn:
#                 # Get user statistics
#                 user_stats = pd.read_sql("""
#                     SELECT 
#                         COUNT(*) as total_users,
#                         SUM(image_count) as total_images,
#                         AVG(image_count) as avg_images_per_user
#                     FROM users
#                 """, conn)
                
#                 # Get training statistics
#                 training_stats = pd.read_sql("""
#                     SELECT 
#                         COUNT(*) as total_sessions,
#                         AVG(accuracy) as avg_accuracy,
#                         MAX(accuracy) as best_accuracy
#                     FROM training_sessions
#                 """, conn)
                
#                 st.write(f"Total Users: {user_stats['total_users'].iloc[0]}")
#                 st.write(f"Total Images: {user_stats['total_images'].iloc[0]}")
#                 st.write(f"Avg Images/User: {user_stats['avg_images_per_user'].iloc[0]:.1f}")
#                 st.write(f"Training Sessions: {training_stats['total_sessions'].iloc[0]}")
#                 if training_stats['total_sessions'].iloc[0] > 0:
#                     st.write(f"Average Accuracy: {training_stats['avg_accuracy'].iloc[0]:.2%}")
#                     st.write(f"Best Accuracy: {training_stats['best_accuracy'].iloc[0]:.2%}")
        
#         except Exception as e:
#             st.error(f"Error accessing database: {str(e)}")
    
#     # Training History
#     st.subheader("Training History")
#     try:
#         with sqlite3.connect(st.session_state.system.db_path) as conn:
#             training_history = pd.read_sql("""
#                 SELECT 
#                     timestamp,
#                     num_users,
#                     accuracy,
#                     metrics
#                 FROM training_sessions 
#                 ORDER BY timestamp DESC
#             """, conn)
            
#             if not training_history.empty:
#                 # Create accuracy trend plot
#                 fig, ax = plt.subplots(figsize=(10, 4))
#                 ax.plot(range(len(training_history)), training_history['accuracy'], marker='o')
#                 ax.set_xlabel('Training Session')
#                 ax.set_ylabel('Accuracy')
#                 ax.set_title('Model Accuracy Trend')
#                 st.pyplot(fig)
                
#                 # Display detailed history
#                 st.dataframe(training_history)
#             else:
#                 st.info("No training history available")
    
#     except Exception as e:
#         st.error(f"Error loading training history: {str(e)}")
    
#     # System Logs
#     st.subheader("System Logs")
#     show_logs = st.checkbox("Show System Logs")
#     if show_logs:
#         try:
#             log_path = Path('face_recognition.log')
#             if log_path.exists():
#                 with open(log_path, 'r') as f:
#                     logs = f.readlines()
#                 st.code(''.join(logs[-50:]))  # Show last 50 lines
#             else:
#                 st.info("No log file found")
#         except Exception as e:
#             st.error(f"Error reading log file: {str(e)}")
    
#     # System Maintenance
#     st.subheader("System Maintenance")
#     col1, col2 = st.columns(2)
    
#     with col1:
#         if st.button("Clear Image Cache"):
#             try:
#                 # Remove all images but keep the directory
#                 for img_file in st.session_state.system.image_dir.glob("*.jpg"):
#                     img_file.unlink()
#                 st.success("Image cache cleared successfully")
                
#                 # Update database
#                 with sqlite3.connect(st.session_state.system.db_path) as conn:
#                     cursor = conn.cursor()
#                     cursor.execute("UPDATE users SET image_count = 0")
#                     conn.commit()
#             except Exception as e:
#                 st.error(f"Error clearing cache: {str(e)}")
    
#     with col2:
#         if st.button("Reset System"):
#             try:
#                 # Clear all data
#                 for directory in [st.session_state.system.image_dir, 
#                                 st.session_state.system.model_dir]:
#                     for file in directory.glob("*"):
#                         file.unlink()
                
#                 # Reset database
#                 with sqlite3.connect(st.session_state.system.db_path) as conn:
#                     cursor = conn.cursor()
#                     cursor.execute("DELETE FROM users")
#                     cursor.execute("DELETE FROM training_sessions")
#                     conn.commit()
                
#                 st.success("System reset successfully")
#                 st.info("Please restart the application")
#             except Exception as e:
#                 st.error(f"Error resetting system: {str(e)}")

# if __name__ == "__main__":

#     # Configure logging
#     logging.basicConfig(
#         filename='face_recognition.log',
#         level=logging.INFO,
#         format='%(asctime)s - %(levelname)s - %(message)s'
#     )
    
#     # Handle uncaught exceptions
#     def handle_exception(exc_type, exc_value, exc_traceback):
#         if issubclass(exc_type, KeyboardInterrupt):
#             sys.__excepthook__(exc_type, exc_value, exc_traceback)
#             return
#         logger.error("Uncaught exception", exc_info=(exc_type, exc_value, exc_traceback))
    
#     sys.excepthook = handle_exception
    
#     # Run the application
#     main()





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
                
                # # Drop existing tables if they exist (for schema update)
                # cursor.execute("DROP TABLE IF EXISTS recognition_logs")
                # cursor.execute("DROP TABLE IF EXISTS recognition_sessions")
                # cursor.execute("DROP TABLE IF EXISTS training_sessions")
                # cursor.execute("DROP TABLE IF EXISTS users")
                
                # Users table
                cursor.execute('''
                    CREATE TABLE users (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        name TEXT UNIQUE NOT NULL,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        image_count INTEGER DEFAULT 0,
                        last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        status TEXT DEFAULT 'active'
                    )
                ''')
                
                # Training sessions table
                cursor.execute('''
                    CREATE TABLE training_sessions (
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
                
                # Recognition sessions table
                cursor.execute('''
                    CREATE TABLE recognition_sessions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                        user_id INTEGER,
                        confidence REAL,
                        duration REAL,
                        status TEXT DEFAULT 'active',
                        FOREIGN KEY(user_id) REFERENCES users(id)
                    )
                ''')
                
                # Recognition logs table
                cursor.execute('''
                    CREATE TABLE recognition_logs (
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

class FaceRecognitionSystem:
    """Main face recognition system class"""
    def __init__(self):
        """Initialize the face recognition system"""
        self.setup_directories()
        self.config = ConfigManager()
        self.db = DatabaseManager(self.base_dir / 'face_recognition.db')
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
            # Continuing from previous code...

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
        """Train face recognition model with progress tracking"""
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
                    username = img_path.stem.split('_')[0]
                    user_id = int(img_path.stem.split('_')[1])
                    
                    image = cv2.imread(str(img_path))
                    if image is None:
                        continue

                    face, _ = self.detect_face(image)
                    if face is not None:
                        face_features = self.extract_features(face)
                        if face_features is not None:
                            features.append(face_features)
                            labels.append(user_id)
                            user_images[user_id] = user_images.get(user_id, 0) + 1
                    
                    if progress_callback:
                        progress_callback(idx + 1, total_images)

                except Exception as e:
                    logger.error(f"Error processing image {img_path}: {e}")
                    continue

            if len(user_images) < 2:
                raise ValueError(f"Need at least 2 different users for training. Current users: {len(user_images)}")

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

def init_session_state():
    """Initialize session state variables"""
    if 'system_initialized' not in st.session_state:
        try:
            st.session_state.system = FaceRecognitionSystem()
            st.session_state.system_initialized = True
            st.session_state.camera_active = False
            st.session_state.recognition_active = False
            st.session_state.current_user = None
            st.session_state.recognition_session_id = None
            logger.info("Session state initialized successfully")
        except Exception as e:
            logger.error(f"Error initializing session state: {e}")
            st.error(f"Error initializing system: {str(e)}")

def show_dashboard():
    """Display dashboard with system statistics"""
    st.title("Face Recognition Dashboard")
    
    try:
        with sqlite3.connect(st.session_state.system.db.db_path) as conn:
            # Get system stats
            users_df = pd.read_sql("""
                SELECT COUNT(*) as total_users, 
                    SUM(image_count) as total_images,
                    MAX(last_updated) as last_update
                FROM users
            """, conn)
            
            training_df = pd.read_sql("""
                SELECT accuracy, timestamp 
                FROM training_sessions 
                ORDER BY timestamp DESC LIMIT 1
            """, conn)
            
            recognition_df = pd.read_sql("""
                SELECT COUNT(*) as total_recognitions,
                    AVG(confidence) as avg_confidence
                FROM recognition_logs
                WHERE timestamp >= datetime('now', '-24 hours')
            """, conn)
            
            # Display metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Registered Users", int(users_df['total_users'].iloc[0]))
            with col2:
                st.metric("Total Images", int(users_df['total_images'].iloc[0]))
            with col3:
                if not training_df.empty:
                    st.metric("Latest Accuracy", f"{training_df['accuracy'].iloc[0]:.2%}")
                else:
                    st.metric("Latest Accuracy", "No data")
            with col4:
                st.metric("24h Recognitions", int(recognition_df['total_recognitions'].iloc[0]))

            # Recent activity
            st.subheader("Recent Activity")
            recent_activity = pd.read_sql("""
                SELECT u.name, r.timestamp, r.confidence
                FROM recognition_logs r
                JOIN users u ON r.user_id = u.id
                ORDER BY r.timestamp DESC
                LIMIT 10
            """, conn)
            
            if not recent_activity.empty:
                st.dataframe(recent_activity)
            else:
                st.info("No recent activity")

            # Training history
            st.subheader("Training History")
            training_history = pd.read_sql("""
                SELECT timestamp, accuracy, precision_score, recall_score, f1_score
                FROM training_sessions
                ORDER BY timestamp DESC
                LIMIT 5
            """, conn)
            
            if not training_history.empty:
                st.line_chart(training_history.set_index('timestamp')[['accuracy', 'precision_score', 'recall_score', 'f1_score']])
            else:
                st.info("No training history available")

    except Exception as e:
        logger.error(f"Error displaying dashboard: {e}")
        st.error("Error loading dashboard data")

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
    """Face recognition interface"""
    st.title("Face Recognition")

    if not hasattr(st.session_state.system, 'classifier') or st.session_state.system.classifier is None:
        st.warning("No trained model available. Please train the model first.")
        return

    try:
        # Create two columns for the interface
        col1, col2 = st.columns([2, 1])
        
        with col1:
            # Camera feed placeholder
            video_placeholder = st.empty()
            status_placeholder = st.empty()
            
            # Control buttons
            button_col1, button_col2 = st.columns(2)
            start_button = button_col1.button("Start Recognition")
            stop_button = button_col2.button("Stop Recognition")

            if start_button:
                st.session_state.recognition_active = True
                st.session_state.recognition_session_id = st.session_state.system.db.start_recognition_session()
            if stop_button:
                st.session_state.recognition_active = False
                if hasattr(st.session_state, 'recognition_session_id'):
                    st.session_state.system.db.close_recognition_session(st.session_state.recognition_session_id)

        with col2:
            # Recognition Logs Section
            st.subheader("Recognition Logs")
            
            # Get recent recognitions
            with sqlite3.connect(st.session_state.system.db.db_path) as conn:
                logs_df = pd.read_sql("""
                    SELECT 
                        r.timestamp,
                        u.name as recognized_person,
                        r.confidence
                    FROM recognition_logs r
                    JOIN users u ON r.user_id = u.id
                    ORDER BY r.timestamp DESC
                    LIMIT 10
                """, conn)
                
                if not logs_df.empty:
                    # Format the dataframe for display
                    logs_df['timestamp'] = pd.to_datetime(logs_df['timestamp']).dt.strftime('%H:%M:%S')
                    logs_df['confidence'] = logs_df['confidence'].apply(lambda x: f"{x:.1%}")
                    st.dataframe(logs_df, use_container_width=True)
                else:
                    st.info("No recognition logs yet")
            
            # Statistics
            st.subheader("Session Statistics")
            if hasattr(st.session_state, 'recognition_session_id'):
                stats = st.session_state.system.db.get_recognition_stats(hours=1)  # Last hour stats
                st.metric("Recognitions (Last Hour)", stats['total_recognitions'])
                if stats['avg_confidence']:
                    st.metric("Average Confidence", f"{stats['avg_confidence']:.1%}")
                st.metric("Unique Users Recognized", stats['unique_users'])

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

                            if confidence > 0.5:
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
    
    # System status indicator
    if st.session_state.system_initialized:
        st.sidebar.success("System Status: Active")
    else:
        st.sidebar.error("System Status: Inactive")
    
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

if __name__ == "__main__":
    main()