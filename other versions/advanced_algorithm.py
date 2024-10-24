import os
import cv2
import numpy as np
import sqlite3
import json
import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Input, Lambda
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, roc_curve, auc
from sklearn.svm import SVC
from sklearn.ensemble import VotingClassifier
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import time
import pickle
from mtcnn import MTCNN
from retinaface import RetinaFace
from sklearn.preprocessing import LabelEncoder

class AdvancedFaceRecognitionSystem:
    def __init__(self, input_shape=(224, 224, 3), n_classes=None, face_detection_method='mtcnn'):
        self.input_shape = input_shape
        self.n_classes = n_classes
        self.face_detection_method = face_detection_method
        self.face_detector = self.initialize_face_detector()
        self.models = None
        self.ensemble = None
        self.data_generator = None
        self.db_path = "face_recognition.db"
        self.results_dir = "results"
        if not os.path.exists(self.results_dir):
            os.makedirs(self.results_dir)
        self.initialize_database()

    def initialize_database(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create users table
            cursor.execute('''CREATE TABLE IF NOT EXISTS users
                              (id INTEGER PRIMARY KEY AUTOINCREMENT,
                               username TEXT UNIQUE NOT NULL)''')
            
            # Create training_sessions table with user_id column
            cursor.execute('''CREATE TABLE IF NOT EXISTS training_sessions
                              (id INTEGER PRIMARY KEY AUTOINCREMENT,
                               user_id INTEGER,
                               timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                               num_users INTEGER,
                               num_images INTEGER,
                               model_accuracy REAL,
                               training_stats TEXT,
                               FOREIGN KEY (user_id) REFERENCES users(id))''')
            
            # Create recognition_sessions table
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

    def build_models(self):
        # Build and return a dictionary of models
        models = {
            'resnet50': self.build_transfer_model(ResNet50, 'resnet50'),
            'siamese': self.build_siamese_model()
        }
        return models

    def build_transfer_model(self, base_model_class, name):
        # Build a transfer learning model using the specified base model
        base_model = base_model_class(weights='imagenet', include_top=False, input_shape=self.input_shape)
        x = base_model.output
        x = GlobalAveragePooling2D()(x)
        x = Dense(1024, activation='relu')(x)
        output = Dense(self.n_classes, activation='softmax')(x)
        model = Model(inputs=base_model.input, outputs=output, name=name)
        
        # Freeze base model layers
        for layer in base_model.layers:
            layer.trainable = False
        
        model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])
        return model

    def build_siamese_model(self):
        # Build a Siamese network for face comparison
        base_network = self.get_base_network()
        input_a = Input(shape=self.input_shape)
        input_b = Input(shape=self.input_shape)
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)
        distance = Lambda(lambda x: tf.keras.backend.abs(x[0] - x[1]))([processed_a, processed_b])
        output = Dense(1, activation='sigmoid')(distance)
        model = Model(inputs=[input_a, input_b], outputs=output)
        model.compile(loss='binary_crossentropy', optimizer=Adam(learning_rate=0.001), metrics=['accuracy'])
        return model

    def get_base_network(self):
        # Define the base network for the Siamese model
        input = Input(shape=self.input_shape)
        x = tf.keras.layers.Conv2D(32, (7, 7), activation='relu')(input)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = tf.keras.layers.MaxPooling2D()(x)
        x = tf.keras.layers.Conv2D(64, (3, 3), activation='relu')(x)
        x = GlobalAveragePooling2D()(x)
        x = Dense(128, activation='relu')(x)
        return Model(inputs=input, outputs=x)

    def build_ensemble(self):
        # Build an ensemble of models for improved prediction
        estimators = [
            ('resnet50', self.models['resnet50']),
            ('svm', SVC(kernel='rbf', probability=True))
        ]
        ensemble = VotingClassifier(estimators=estimators, voting='soft')
        return ensemble

    def build_data_generator(self):
        # Create a data generator for data augmentation
        return ImageDataGenerator(
            rotation_range=20,
            width_shift_range=0.2,
            height_shift_range=0.2,
            horizontal_flip=True,
            zoom_range=0.2,
            brightness_range=[0.8, 1.2]  # Added brightness augmentation
        )

    def initialize_face_detector(self):
        if self.face_detection_method == 'haarcascade':
            return cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        elif self.face_detection_method == 'mtcnn':
            return MTCNN()
        elif self.face_detection_method == 'retinaface':
            return RetinaFace()
        else:
            raise ValueError("Invalid face detection method. Choose 'haarcascade', 'mtcnn', or 'retinaface'.")

    def preprocess_image(self, image):
        if self.face_detection_method == 'haarcascade':
            return self.preprocess_image_haarcascade(image)
        elif self.face_detection_method == 'mtcnn':
            return self.preprocess_image_mtcnn(image)
        elif self.face_detection_method == 'retinaface':
            return self.preprocess_image_retinaface(image)

    def preprocess_image_haarcascade(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        faces = self.face_detector.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)
        
        if len(faces) == 0:
            return None
        
        (x, y, w, h) = faces[0]
        face = gray[y:y+h, x:x+w]
        face = cv2.resize(face, (224, 224))
        face = cv2.equalizeHist(face)
        
        return (face, (x, y, w, h))

    def preprocess_image_mtcnn(self, image):
        faces = self.face_detector.detect_faces(image)
        
        if not faces:
            return None
        
        face = faces[0]
        x, y, w, h = face['box']
        face_img = image[y:y+h, x:x+w]
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = cv2.resize(face_img, (224, 224))
        face_img = cv2.equalizeHist(face_img)
        
        return (face_img, (x, y, w, h))

    def preprocess_image_retinaface(self, image):
        faces = self.face_detector.detect_faces(image)
        
        if not faces:
            return None
        
        face = list(faces.values())[0]
        x1, y1, x2, y2 = face['facial_area']
        w, h = x2 - x1, y2 - y1
        face_img = image[y1:y2, x1:x2]
        face_img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        face_img = cv2.resize(face_img, (224, 224))
        face_img = cv2.equalizeHist(face_img)
        
        return (face_img, (x1, y1, w, h))

    def extract_features(self, face):
        # Extract features using SIFT and deep learning
        keypoints, descriptors = self.sift.detectAndCompute(face, None)
        if descriptors is not None:
            sift_features = np.mean(descriptors, axis=0)
        else:
            sift_features = np.zeros(128)
        
        face_rgb = cv2.cvtColor(face, cv2.COLOR_GRAY2RGB)
        face_rgb = cv2.resize(face_rgb, (224, 224))
        face_rgb = np.expand_dims(face_rgb, axis=0)
        face_rgb = tf.keras.applications.resnet50.preprocess_input(face_rgb)
        deep_features = self.models['resnet50'].predict(face_rgb).flatten()
        
        combined_features = np.concatenate([sift_features, deep_features])
        
        return combined_features

    def generate_dataset(self):
        if not os.path.exists('image_data'):
            os.makedirs('image_data')

        username = input("Enter the person's name: ")
        user_id = self.add_user_to_db(username)
        
        cap = cv2.VideoCapture(0)
        frame_count = 0
        total_imgs = 20

        while frame_count < total_imgs:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera.")
                continue

            result = self.preprocess_image(frame)
            if result is not None:
                face, (x, y, w, h) = result
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                
                frame_count += 1
                img_name = f"image_data/{username}.{user_id}.{frame_count}.jpg"
                cv2.imwrite(img_name, face)
                cv2.putText(frame, f"Captured {frame_count}/{total_imgs}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                print(f"Captured image {frame_count}/{total_imgs}: {img_name}")
            else:
                cv2.putText(frame, "No face detected", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.imshow('Capturing Face Data', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()
        print(f"Dataset generation completed for {username}. {frame_count} images captured.")
        self.update_training_session(user_id, frame_count)

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
            """, (user_id, num_images, accuracy, json.dumps(training_stats) if training_stats else None))
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        finally:
            if conn:
                conn.close()

    def train(self, X, y, epochs=20, batch_size=32):
        num_classes = len(np.unique(y))
        y_categorical = to_categorical(y, num_classes=num_classes)

        if num_classes != self.n_classes:
            print(f"Adjusting model for {num_classes} classes")
            self.n_classes = num_classes
            self.models = self.build_models()
            self.ensemble = self.build_ensemble()
            self.data_generator = self.build_data_generator()

        X_train, X_val, y_train, y_val = train_test_split(X, y_categorical, test_size=0.2, random_state=42)

        training_start_time = time.time()

        for name, model in self.models.items():
            if name != 'siamese':
                print(f"Training {name} model...")
                model.fit(self.data_generator.flow(X_train, y_train, batch_size=batch_size),
                          validation_data=(X_val, y_val),
                          epochs=epochs,
                          callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

        print("Training Siamese model...")
        self.train_siamese(X_train, np.argmax(y_train, axis=1), epochs, batch_size)

        print("Training SVM...")
        resnet_features = self.models['resnet50'].predict(X_train)
        self.ensemble.estimators_[1][1].fit(resnet_features, np.argmax(y_train, axis=1))

        print("Training ensemble...")
        ensemble_features = np.hstack([model.predict(X_train) for name, model in self.models.items() if name != 'siamese'])
        self.ensemble.fit(ensemble_features, np.argmax(y_train, axis=1))

        training_end_time = time.time()
        training_duration = training_end_time - training_start_time

        # Evaluate the model
        y_pred = self.predict(X_val)
        accuracy = accuracy_score(np.argmax(y_val, axis=1), y_pred)
        precision = precision_score(np.argmax(y_val, axis=1), y_pred, average='weighted')
        recall = recall_score(np.argmax(y_val, axis=1), y_pred, average='weighted')
        f1 = f1_score(np.argmax(y_val, axis=1), y_pred, average='weighted')

        training_stats = {
            "num_classes": num_classes,
            "total_images": len(X),
            "training_duration": float(training_duration),
            "accuracy": float(accuracy),
            "precision": float(precision),
            "recall": float(recall),
            "f1_score": float(f1)
        }

        print(f"Training completed. Accuracy: {accuracy:.2f}")
        print(f"Training stats: {training_stats}")

        self.save_training_stats_visual(training_stats, np.argmax(y_val, axis=1), y_pred)
        self.update_training_session(None, len(X), accuracy, training_stats)

    def train_siamese(self, X, y, epochs, batch_size):
        def generate_pairs(X, y):
            pairs = []
            labels = []
            n_classes = len(np.unique(y))
            for i in range(len(X)):
                positive_idx = np.random.choice(np.where(y == y[i])[0])
                negative_idx = np.random.choice(np.where(y != y[i])[0])
                pairs += [[X[i], X[positive_idx]], [X[i], X[negative_idx]]]
                labels += [1, 0]
            return np.array(pairs), np.array(labels)

        pairs, pair_labels = generate_pairs(X, y)
        self.models['siamese'].fit([pairs[:, 0], pairs[:, 1]], pair_labels, epochs=epochs, batch_size=batch_size,
                                   callbacks=[tf.keras.callbacks.EarlyStopping(patience=5, restore_best_weights=True)])

    def predict(self, X):
        ensemble_features = np.hstack([model.predict(X) for name, model in self.models.items() if name != 'siamese'])
        return self.ensemble.predict(ensemble_features)

    def recognize_face(self, frame):
        result = self.preprocess_image(frame)
        if result is None:
            return None, None, None

        face, (x, y, w, h) = result
        features = self.extract_features(face)
        features = features.reshape(1, -1)
        prediction = self.predict(features)
        predicted_class = np.argmax(prediction)
        confidence = np.max(prediction)

        return predicted_class, confidence, (x, y, w, h)

    def run_recognition(self):
        cap = cv2.VideoCapture(0)
        recognition_stats = {
            "total_frames": 0,
            "faces_detected": 0,
            "faces_recognized": 0,
            "confidence_scores": []
        }
        
        start_time = time.time()
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera.")
                break

            result = self.recognize_face(frame)
            if result is not None:
                predicted_class, confidence, (x, y, w, h) = result
                
                recognition_stats["faces_detected"] += 1
                recognition_stats["confidence_scores"].append(confidence)

                if confidence > 0.5:
                    recognition_stats["faces_recognized"] += 1
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                    cv2.putText(frame, f"ID: {predicted_class} ({confidence:.2f})", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
                else:
                    cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                    cv2.putText(frame, "Unknown", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
            else:
                cv2.putText(frame, f"No face detected ({self.face_detection_method})", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)

            cv2.imshow('Face Recognition', frame)
            recognition_stats["total_frames"] += 1

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        end_time = time.time()
        cap.release()
        cv2.destroyAllWindows()

        duration = end_time - start_time
        fps = recognition_stats["total_frames"] / duration
        recognition_rate = recognition_stats["faces_recognized"] / recognition_stats["faces_detected"] if recognition_stats["faces_detected"] > 0 else 0
        avg_confidence = np.mean(recognition_stats["confidence_scores"]) if recognition_stats["confidence_scores"] else 0
        
        print(f"Recognition stats: {recognition_stats}")
        print(f"FPS: {fps:.2f}")
        print(f"Recognition rate: {recognition_rate:.2f}")
        print(f"Average confidence: {avg_confidence:.2f}")

        self.save_recognition_stats(recognition_stats, fps, recognition_rate, avg_confidence)

    def save_recognition_stats(self, stats, fps, recognition_rate, avg_confidence):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO recognition_sessions 
                (total_frames, faces_detected, faces_recognized, recognition_rate, avg_confidence, fps) 
                VALUES (?, ?, ?, ?, ?, ?)
            """, (stats["total_frames"], stats["faces_detected"], 
                  stats["faces_recognized"], recognition_rate, avg_confidence, fps))
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        finally:
            if conn:
                conn.close()

    def save_training_stats_visual(self, training_stats, y_true, y_pred):
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

        # Confusion Matrix Heatmap
        cm = confusion_matrix(y_true, y_pred)
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.savefig(os.path.join(self.results_dir, 'confusion_matrix.png'))
        plt.close()

        # ROC Curve (for binary classification)
        if len(np.unique(y_true)) == 2:
            fpr, tpr, _ = roc_curve(y_true, self.ensemble.predict_proba(X_test)[:, 1])
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

        # Save JSON file
        with open(os.path.join(self.results_dir, 'training_stats.json'), 'w') as f:
            json.dump(training_stats, f, indent=4)

        print(f"Training stats visualizations and JSON saved in {self.results_dir}")

    def save_model(self, filename='face_recognition_model.pkl'):
        with open(filename, 'wb') as file:
            pickle.dump({
                'models': self.models,
                'ensemble': self.ensemble,
                'n_classes': self.n_classes
            }, file)
        print(f"Model saved to {filename}")

    def load_model(self, filename='face_recognition_model.pkl'):
        try:
            with open(filename, 'rb') as file:
                data = pickle.load(file)
                self.models = data['models']
                self.ensemble = data['ensemble']
                self.n_classes = data['n_classes']
            print(f"Model loaded from {filename}")
            return True
        except FileNotFoundError:
            print(f"Model file {filename} not found.")
        except Exception as e:
            print(f"Error loading model: {e}")
        return False

    def save_training_stats(self, training_stats):
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                INSERT INTO training_sessions 
                (user_id, num_users, num_images, model_accuracy, training_stats) 
                VALUES (?, ?, ?, ?, ?)
            """, (1, training_stats["num_classes"], training_stats["num_images"], 
                  training_stats["accuracy"], json.dumps(training_stats)))
            conn.commit()
        except sqlite3.Error as e:
            print(f"Database error: {e}")
        finally:
            if conn:
                conn.close()

def load_dataset(dataset_path='image_data'):
    X = []
    y = []
    for person_id in os.listdir(dataset_path):
        person_dir = os.path.join(dataset_path, person_id)
        if os.path.isdir(person_dir):
            for image_file in os.listdir(person_dir):
                image_path = os.path.join(person_dir, image_file)
                image = cv2.imread(image_path)
                if image is not None:
                    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
                    resized = cv2.resize(gray, (224, 224))
                    X.append(resized)
                    y.append(person_id)
    
    X = np.array(X)
    X = X.reshape(X.shape[0], 224, 224, 1)  # Reshape for the CNN
    X = X / 255.0  # Normalize pixel values
    
    # Encode labels
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    return X, y, le.classes_

def main():
    print("Choose a face detection method:")
    print("1. Haar Cascade")
    print("2. MTCNN")
    print("3. RetinaFace")
    choice = input("Enter your choice (1-3): ")

    if choice == '1':
        face_detection_method = 'haarcascade'
    elif choice == '2':
        face_detection_method = 'mtcnn'
    elif choice == '3':
        face_detection_method = 'retinaface'
    else:
        print("Invalid choice. Using MTCNN as default.")
        face_detection_method = 'mtcnn'

    system = AdvancedFaceRecognitionSystem(face_detection_method=face_detection_method)
    model_filename = 'face_recognition_model.pkl'

    if os.path.exists(model_filename):
        load_choice = input(f"A trained model ({model_filename}) is available. Do you want to load it? (y/n): ").lower()
        if load_choice == 'y':
            if system.load_model(model_filename):
                print("Model loaded successfully.")
            else:
                print("Failed to load the model. Proceeding with the main menu.")
        else:
            print("Proceeding without loading the model.")
    else:
        print("No trained model found. You'll need to train a new model.")

    while True:
        print("\nAdvanced Face Recognition System Menu:")
        print("1. Generate Dataset")
        print("2. Train/Retrain Classifier")
        print("3. Run Face Recognition")
        print("4. Save Model")
        print("5. Exit")
        
        choice = input("Enter your choice (1-5): ")
        
        if choice == '1':
            num_people = int(input("How many people do you want to add to the dataset? "))
            for _ in range(num_people):
                system.generate_dataset()
        elif choice == '2':
            try:
                X, y, class_names = load_dataset()
                if len(np.unique(y)) < 2:
                    print("Error: Dataset must contain at least two different classes.")
                else:
                    system.n_classes = len(np.unique(y))
                    system.train(X, y)
            except Exception as e:
                print(f"Error during training: {e}")
        elif choice == '3':
            if system.models is None:
                print("No trained model available. Please train the classifier first.")
            else:
                system.run_recognition()
        elif choice == '4':
            system.save_model()
        elif choice == '5':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()