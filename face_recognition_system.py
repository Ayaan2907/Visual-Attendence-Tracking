import os
import cv2
import numpy as np
import tensorflow as tf
import pickle
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing import image as keras_image
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
import matplotlib.pyplot as plt
import seaborn as sns

from config_manager import ConfigManager
from database_manager import DatabaseManager
from encryption_manager import EncryptionManager

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
        logging.info("Face Recognition System initialized successfully")

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
            
            logging.info("Models initialized successfully")
            
        except Exception as e:
            logging.error(f"Error initializing models: {e}")
            raise

    def load_classifier(self) -> None:
        """Load the trained classifier if it exists"""
        try:
            model_files = list(self.model_dir.glob('*.pkl'))
            if model_files:
                latest_model = max(model_files, key=lambda x: x.stat().st_mtime)
                with open(latest_model, 'rb') as f:
                    self.classifier = pickle.load(f)
                logging.info(f"Loaded classifier from {latest_model}")
            else:
                self.classifier = None
                logging.info("No trained classifier found")
        except Exception as e:
            logging.error(f"Error loading classifier: {e}")
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
            logging.error(f"Error saving model: {e}")
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
            logging.error(f"Error in face detection: {e}")
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
            logging.error(f"Error extracting features: {e}")
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

            logging.info(f"Successfully captured {len(image_paths)} images for user {username}")
            return image_paths

        except Exception as e:
            logging.error(f"Error capturing dataset: {e}")
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
            logging.error(f"Error training model: {e}")
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
            logging.error(f"Error in face recognition: {e}")
            return None, None, None
