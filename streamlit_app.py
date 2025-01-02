import streamlit as st
import pandas as pd
import cv2
import numpy as np
import time
import logging
from pathlib import Path
from typing import Optional, Tuple, List, Dict, Any
from datetime import datetime

from face_recognition_system import FaceRecognitionSystem
from database_manager import DatabaseManager
from encryption_manager import EncryptionManager
from config_manager import ConfigManager

# Initialize system components
face_recognition_system = FaceRecognitionSystem()
db_manager = face_recognition_system.db
encryption_manager = face_recognition_system.encryption_manager
config_manager = face_recognition_system.config

def show_dataset_collection():
    st.title("Dataset Collection")
    username = st.text_input("Enter your name:")
    num_images = st.slider("Number of images to capture:", min_value=10, max_value=100, value=20)
    
    if st.button("Start Collection"):
        if username:
            image_paths = face_recognition_system.capture_dataset(username, num_images)
            st.success(f"Captured {len(image_paths)} images for {username}")
        else:
            st.error("Please enter a valid name")

def show_model_training():
    st.title("Model Training")
    
    if st.button("Start Training"):
        metrics = face_recognition_system.train_model()
        st.success("Model trained successfully")
        st.json(metrics)

def show_recognition():
    st.title("Face Recognition")
    st.write("Press 'Start Recognition' to begin")
    
    if st.button("Start Recognition"):
        cap = cv2.VideoCapture(0)
        if not cap.isOpened():
            st.error("Failed to open camera")
            return
        
        session_id = db_manager.start_recognition_session()
        st.session_state['session_id'] = session_id
        
        while True:
            ret, frame = cap.read()
            if not ret:
                st.error("Failed to capture frame")
                break
            
            user_id, confidence, bbox = face_recognition_system.recognize_face(frame)
            if user_id is not None:
                user_info = db_manager.get_user(user_id)
                st.write(f"Recognized: {user_info['name']} with confidence {confidence:.2f}")
                db_manager.log_recognition(session_id, user_id, confidence)
            else:
                st.write("No face recognized")
            
            time.sleep(0.1)
        
        cap.release()
        db_manager.close_recognition_session(session_id)

def show_settings():
    st.title("Settings")
    st.write("Configure system settings")
    
    config = config_manager.config
    st.json(config)
    
    if st.button("Save Settings"):
        config_manager.save_config()
        st.success("Settings saved successfully")

def show_system_logs():
    st.title("System Logs")
    log_files = list(Path('face_recognition_data/logs').glob('*.log'))
    
    if log_files:
        latest_log = max(log_files, key=lambda x: x.stat().st_mtime)
        with open(latest_log, 'r') as f:
            logs = f.read()
        st.text_area("Logs", logs, height=400)
    else:
        st.write("No logs available")

def init_session_state():
    if 'session_id' not in st.session_state:
        st.session_state['session_id'] = None

def main():
    st.sidebar.title("Navigation")
    options = ["Dataset Collection", "Model Training", "Face Recognition", "Settings", "System Logs"]
    choice = st.sidebar.radio("Go to", options)
    
    init_session_state()
    
    if choice == "Dataset Collection":
        show_dataset_collection()
    elif choice == "Model Training":
        show_model_training()
    elif choice == "Face Recognition":
        show_recognition()
    elif choice == "Settings":
        show_settings()
    elif choice == "System Logs":
        show_system_logs()

def authenticate(username: str, password: str) -> bool:
    user = db_manager.authenticate_user(username, password)
    if user:
        st.session_state['user'] = user
        return True
    return False

def show_dashboard():
    st.title("Dashboard")
    st.write(f"Welcome, {st.session_state['user']['name']}")
    
    if st.button("Logout"):
        logout()

def logout():
    st.session_state['user'] = None
    st.experimental_rerun()

if __name__ == "__main__":
    if 'user' not in st.session_state:
        st.session_state['user'] = None
    
    if st.session_state['user'] is None:
        st.title("Login")
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        
        if st.button("Login"):
            if authenticate(username, password):
                st.success("Login successful")
                st.experimental_rerun()
            else:
                st.error("Invalid username or password")
    else:
        show_dashboard()
        main()
