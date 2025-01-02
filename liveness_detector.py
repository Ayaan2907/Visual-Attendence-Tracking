import cv2
import dlib
import numpy as np
import tensorflow as tf
from collections import deque
from scipy.spatial import distance
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout
from skimage.feature import local_binary_pattern
import logging

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
        
        logging.info("Liveness detector initialized successfully")

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
                logging.info("Loaded pre-trained liveness detection weights")
            except:
                logging.warning("No pre-trained weights found for liveness detection")
            
        except Exception as e:
            logging.error(f"Error initializing liveness detection models: {e}")
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
            logging.error(f"Error in blink detection: {e}")
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
            logging.error(f"Error in depth analysis: {e}")
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
            logging.error(f"Error in head movement detection: {e}")
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
            logging.error(f"Error in texture analysis: {e}")
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
            logging.error(f"Error in liveness detection: {e}")
            return False, {}, 0.0
