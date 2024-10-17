import cv2
import numpy as np
import os
import sqlite3
import json
from PIL import Image
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.svm import SVC
import time

class FaceTrainer:
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
        self.sift = cv2.SIFT_create()
        self.mobilenet_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224, 224, 3), pooling='avg')
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.db_path = "face_recognition.db"
        self.initialize_database()

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
                               user_id INTEGER,
                               timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                               num_images INTEGER,
                               model_accuracy REAL,
                               training_stats TEXT,
                               FOREIGN KEY (user_id) REFERENCES users(id))''')
            
            conn.commit()
            print("Database initialized successfully")
        except sqlite3.Error as e:
            print(f"Database error: {e}")
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

    def generate_dataset(self):
        if not os.path.exists('data'):
            os.makedirs('data')

        username = input("Enter the person's name: ")
        
        # Add user to the database if not exists
        user_id = self.add_user_to_db(username)
        
        cap = cv2.VideoCapture(0)
        img_id = 0
        total_imgs = 20

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera.")
                break

            result = self.preprocess_image(frame)
            if result is not None:
                face, (x, y, w, h) = result
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                img_id += 1
                img_name = f"data/{username}.{user_id}.{img_id}.jpg"
                cv2.imwrite(img_name, face)
                cv2.putText(frame, f"Captured {img_id}/{total_imgs}", (50, 50), self.font, 0.9, (0, 255, 0), 2)
                print(f"Captured image {img_id}/{total_imgs}: {img_name}")
                
                time.sleep(0.25)
            else:
                cv2.putText(frame, "No face detected", (50, 50), self.font, 0.9, (0, 0, 255), 2)

            cv2.imshow('Capturing Face Data', frame)

            if cv2.waitKey(1) & 0xFF == ord('q') or img_id == total_imgs:
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"Dataset generation completed for {username}. {img_id} images captured.")

        # Update training session in the database
        self.update_training_session(user_id, img_id)

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

    def train_classifier(self):
        data_dir = "data"
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

        # Train SVM classifier
        self.svm_classifier = SVC(probability=True)
        self.svm_classifier.fit(X_train, y_train)

        # Evaluate the model
        y_pred = self.svm_classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')

        training_stats = {
            "unique_ids": len(set(ids)),
            "total_images": len(faces),
            "images_per_person": len(faces) / len(set(ids)),
            "accuracy": accuracy,
            "precision": precision,
            "recall": recall,
            "f1_score": f1
        }

        # Save to database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO training_sessions 
                (username, num_images, model_accuracy, training_stats) 
                VALUES (?, ?, ?, ?)
            """, ("latest_training", len(faces), accuracy, json.dumps(training_stats)))
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        finally:
            if conn:
                conn.close()

        print(f"Training completed. Accuracy: {accuracy:.2f}")
        print(f"Training stats: {training_stats}")

    def recognize_face(self):
        cap = cv2.VideoCapture(0)
        recognition_stats = {
            "total_frames": 0,
            "faces_detected": 0,
            "faces_recognized": 0,
            "confidence_scores": []
        }
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera.")
                break

            result = self.preprocess_image(frame)
            if result is not None:
                face, (x, y, w, h) = result
                features = self.extract_features(face)
                
                # Predict using SVM classifier
                prediction = self.svm_classifier.predict([features])
                confidence = self.svm_classifier.predict_proba([features]).max() * 100

                recognition_stats["faces_detected"] += 1
                recognition_stats["confidence_scores"].append(confidence)

                if confidence > 50:
                    recognition_stats["faces_recognized"] += 1
                    label = prediction[0]
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {label} ({confidence:.2f}%)", (x, y-10), self.font, 0.9, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (x, y-10), self.font, 0.9, (0, 0, 255), 2)
            else:
                cv2.putText(frame, "No face detected", (50, 50), self.font, 0.9, (0, 0, 255), 2)

            cv2.imshow('Face Recognition', frame)
            recognition_stats["total_frames"] += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

        recognition_rate = recognition_stats["faces_recognized"] / recognition_stats["faces_detected"] if recognition_stats["faces_detected"] > 0 else 0
        avg_confidence = np.mean(recognition_stats["confidence_scores"]) if recognition_stats["confidence_scores"] else 0
        
        print(f"Recognition stats: {recognition_stats}")
        print(f"Recognition rate: {recognition_rate:.2f}")
        print(f"Average confidence: {avg_confidence:.2f}")

        # Save recognition stats to database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO recognition_sessions 
                (total_frames, faces_detected, faces_recognized, recognition_rate, avg_confidence) 
                VALUES (?, ?, ?, ?, ?)
            """, (recognition_stats["total_frames"], recognition_stats["faces_detected"], 
                  recognition_stats["faces_recognized"], recognition_rate, avg_confidence))
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        finally:
            if conn:
                conn.close()

def main():
    trainer = FaceTrainer()
  
if __name__ == "__main__":
    main()


