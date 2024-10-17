# almost perfect





import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime

class FaceTrainer:
    def __init__(self):
        self.face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.font = cv2.FONT_HERSHEY_SIMPLEX
        self.db_path = "face_recognition.db"
        self.initialize_database()

    def initialize_database(self):
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Create table for training sessions
            cursor.execute('''CREATE TABLE IF NOT EXISTS training_sessions
                              (id INTEGER PRIMARY KEY AUTOINCREMENT,
                               username TEXT NOT NULL,
                               timestamp DATETIME DEFAULT CURRENT_TIMESTAMP,
                               num_images INTEGER,
                               model_accuracy REAL)''')
            
            # Create table for captured images
            cursor.execute('''CREATE TABLE IF NOT EXISTS captured_images
                              (id INTEGER PRIMARY KEY AUTOINCREMENT,
                               session_id INTEGER,
                               image_path TEXT,
                               FOREIGN KEY (session_id) REFERENCES training_sessions(id))''')
            
            conn.commit()
            print("Database initialized successfully")
        except sqlite3.Error as e:
            print(f"Error: {e}")
        finally:
            if conn:
                conn.close()

    def get_username(self):
        username = input("Enter your name: ")
        if not username:
            print("Error: Username is required")
            return None
        return username

    def generate_dataset(self):
        username = self.get_username()
        if not username:
            return

        save_path = "data"
        os.makedirs(save_path, exist_ok=True)

        cap = cv2.VideoCapture(0)
        img_id = 0
        captured_image_paths = []

        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
                roi_gray = gray[y:y+h, x:x+w]
                smiles = self.smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)

                if len(smiles) > 0 and img_id < 30:
                    img_id += 1
                    face = cv2.resize(frame[y:y+h, x:x+w], (450, 450))
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    file_name_path = os.path.join(save_path, f"{username}.{img_id}.jpg")
                    cv2.imwrite(file_name_path, face)
                    cv2.putText(frame, f"Image {img_id}/30", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
                    print(f"Image {img_id} saved at {file_name_path}.")
                    captured_image_paths.append(file_name_path)

            cv2.imshow('Capturing Faces', frame)

            if cv2.waitKey(1) == 13 or img_id == 30:  # Press Enter or capture 30 images to exit
                break

        cap.release()
        cv2.destroyAllWindows()
        print("Dataset generation completed!")

        # Save user information and captured image data to the database
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Insert training session
            cursor.execute("INSERT INTO training_sessions (username, num_images) VALUES (?, ?)", 
                           (username, img_id))
            session_id = cursor.lastrowid
            
            # Insert captured image data
            for image_path in captured_image_paths:
                cursor.execute("INSERT INTO captured_images (session_id, image_path) VALUES (?, ?)",
                               (session_id, image_path))
            
            conn.commit()
            print("User information and image data saved to database")
        except sqlite3.Error as e:
            print(f"Error: {e}")
        finally:
            if conn:
                conn.close()

    def train_classifier(self):
        data_dir = "data"
        path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

        faces = []
        ids = []

        for image in path:
            img = Image.open(image).convert('L')
            imageNp = np.array(img, 'uint8')
            id = int(os.path.split(image)[1].split('.')[1])

            faces.append(imageNp)
            ids.append(id)

        ids = np.array(ids)

        self.recognizer.train(faces, ids)
        self.recognizer.write("classifier.xml")
        
        results_dir = os.path.join(os.getcwd(), "results")
        os.makedirs(results_dir, exist_ok=True)
        
        with open(os.path.join(results_dir, "training_stats.txt"), "w") as f:
            f.write(f"Total images trained: {len(faces)}\n")
            f.write(f"Unique IDs: {len(set(ids))}\n")
            f.write(f"Average images per ID: {len(faces) / len(set(ids)):.2f}\n")
            f.write(f"Minimum images for an ID: {min(np.bincount(ids))}\n")
            f.write(f"Maximum images for an ID: {max(np.bincount(ids))}\n")
            f.write(f"Training date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        
        plt.figure(figsize=(12, 6))
        plt.hist(ids, bins=len(set(ids)), edgecolor='black')
        plt.title("Distribution of Training Images per ID")
        plt.xlabel("ID")
        plt.ylabel("Number of Images")
        plt.xticks(range(min(ids), max(ids)+1))
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        plt.savefig(os.path.join(results_dir, "training_distribution.png"), dpi=300, bbox_inches='tight')
        
        print("Training completed!")
        print(f"Detailed training stats and results saved in /{results_dir}/")
        print(f"Classifier saved as classifier.xml")

    def recognize_face(self):
        cap = cv2.VideoCapture(0)
        
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from camera.")
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = self.face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 255, 0), 2)
                roi_gray = gray[y:y+h, x:x+w]
                roi_gray = cv2.resize(roi_gray, (450, 450))

                id_, confidence = self.recognizer.predict(roi_gray)
                confidence = 100 - int(confidence)
                
                if confidence > 50:
                    try:
                        conn = sqlite3.connect(self.db_path)
                        cursor = conn.cursor()
                        cursor.execute("SELECT username FROM training_sessions WHERE id=?", (id_,))
                        result = cursor.fetchone()
                        
                        if result:
                            username = result[0]
                            cv2.putText(frame, f"{username} ({confidence}%)", (x, y-10), self.font, 0.9, (0, 255, 0), 2)
                        else:
                            cv2.putText(frame, f"Unknown ({confidence}%)", (x, y-10), self.font, 0.9, (0, 255, 0), 2)
                    except sqlite3.Error as e:
                        print(f"Error: {e}")
                    finally:
                        if conn:
                            conn.close()
                else:
                    cv2.putText(frame, "Unknown", (x, y-10), self.font, 0.9, (0, 0, 255), 2)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

def main():
    trainer = FaceTrainer()
    while True:
        print("\nFace Recognition Menu:")
        print("1. Generate Dataset")
        print("2. Train Classifier")
        print("3. Recognize Face")
        print("4. Exit")
        
        choice = input("Enter your choice (1-4): ")
        
        if choice == '1':
            trainer.generate_dataset()
        elif choice == '2':
            trainer.train_classifier()
        elif choice == '3':
            trainer.recognize_face()
        elif choice == '4':
            print("Exiting the program.")
            break
        else:
            print("Invalid choice. Please try again.")

if __name__ == "__main__":
    main()
