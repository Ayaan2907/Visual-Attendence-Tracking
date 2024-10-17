# import cv2

# def just_open_camera():
#     # Open the default camera (usually the built-in webcam)
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         return

#     while True:
#         # Capture frame-by-frame
#         ret, frame = cap.read()

#         if not ret:
#             print("Error: Can't receive frame (stream end?). Exiting ...")
#             break

#         # Display the resulting frame
#         cv2.imshow('Camera Feed', frame)

#         # Press 'q' to quit
#         if cv2.waitKey(1) == ord('q'):
#             break

#     # When everything done, release the capture
#     cap.release()
#     cv2.destroyAllWindows()

# # Call the function to start the camera
# just_open_camera()
# ------------
# import os
# import cv2

# # Load the pre-trained face detection classifier
# face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# def capture_and_save_image(frame):
#     try:
#         # Get the current working directory
#         current_dir = os.getcwd()
#         print(f"Current working directory: {current_dir}")

#         # Create the directory if it doesn't exist
#         save_dir = os.path.join(current_dir, "asset", "images")
#         os.makedirs(save_dir, exist_ok=True)
#         print(f"Save directory: {save_dir}")

#         # Generate a unique filename
#         existing_files = os.listdir(save_dir)
#         file_count = len([f for f in existing_files if f.startswith("captured_image_") and f.endswith(".jpg")])
#         file_name = f"captured_image_{file_count + 1}.jpg"
#         file_path = os.path.join(save_dir, file_name)
        
#         # Save the image
#         success = cv2.imwrite(file_path, frame)
#         if success:
#             print(f"Image successfully saved as {file_path}")
#         else:
#             print(f"Failed to save image to {file_path}")

#     except Exception as e:
#         print(f"An error occurred while saving the image: {str(e)}")

# def face_cropped(img):
#     gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#     if len(faces) == 0:
#         print("No faces detected.")
#         return None

#     for (x, y, w, h) in faces:
#         face_cropped = img[y:y+h, x:x+w]
#         print("Face detected and cropped.")
#         return face_cropped
#     return None

# def open_camera():
#     # Open the default camera (usually the built-in webcam)
#     cap = cv2.VideoCapture(0)

#     if not cap.isOpened():
#         print("Error: Could not open camera.")
#         return

#     while True:
#         # Capture frame-by-frame
#         ret, frame = cap.read()

#         if not ret:
#             print("Error: Can't receive frame (stream end?). Exiting ...")
#             break

#         # Convert frame to grayscale for face detection
#         gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         faces = face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#         # Draw rectangles around detected faces
#         for (x, y, w, h) in faces:
#             cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)  # Red rectangle

#         # Display the resulting frame with face detection
#         cv2.imshow('Camera Feed', frame)
#         # Detect smile
#         smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
        
#         image_count = 0
#         for (x, y, w, h) in faces:
#             roi_gray = gray[y:y+h, x:x+w]
#             smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)
            
#             if len(smiles) > 0 and image_count < 3:
#                 print(f"Smile detected! Capturing image {image_count + 1}...")
#                 capture_and_save_image(frame)
#                 image_count += 1
                
#             if image_count >= 3:
#                 break
        
#         # Press 'q' to quit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break

#     # When everything done, release the capture
#     cap.release()
#     cv2.destroyAllWindows()

# # Call the function to start the camera
# open_camera()
# -----------------
# import cv2
# import os
# import numpy as np
# from PIL import Image
# import matplotlib.pyplot as plt  # Add this import


# class FaceTrainer:
#     def __init__(self):
#         self.face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
#         self.smile_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_smile.xml')
#         self.username = ""

#     def get_username(self):
#         self.username = input("Enter your name: ")
#         if not self.username:
#             print("Error: Username is required")
#             return False
#         return True

#     def face_cropped(self, img):
#         gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#         faces = self.face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#         if len(faces) == 0:
#             return None

#         for (x, y, w, h) in faces:
#             face_cropped = img[y:y+h, x:x+w]
#             return face_cropped
#         return None

#     def generate_dataset(self):
#         if not self.get_username():
#             return

#         save_path = "data"
#         if not os.path.exists(save_path):
#             os.makedirs(save_path)

#         cap = cv2.VideoCapture(0)
#         img_id = 0

#         while True:
#             ret, frame = cap.read()
#             if not ret:
#                 print("Failed to capture frame from camera.")
#                 break

#             gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#             faces = self.face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

#             for (x, y, w, h) in faces:
#                 cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 2)
#                 roi_gray = gray[y:y+h, x:x+w]
#                 smiles = self.smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.8, minNeighbors=20)

#                 if len(smiles) > 0 and img_id < 30:
#                     img_id += 1
#                     face = cv2.resize(frame[y:y+h, x:x+w], (450, 450))
#                     face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
#                     file_name_path = os.path.join(save_path, f"{self.username}.{img_id}.jpg")
#                     cv2.imwrite(file_name_path, face)
#                     cv2.putText(frame, f"Image {img_id}/30", (50, 50), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)
#                     print(f"Image {img_id} saved at {file_name_path}.")

#             cv2.imshow('Capturing Faces', frame)

#             if cv2.waitKey(1) == 13 or img_id == 30:  # Press Enter or capture 30 images to exit
#                 break

#         cap.release()
#         cv2.destroyAllWindows()
#         print("Dataset generation completed!")

#     def train_classifier(self):
#         data_dir = "data"
#         path = [os.path.join(data_dir, f) for f in os.listdir(data_dir)]

#         faces = []
#         ids = []

#         for image in path:
#             img = Image.open(image).convert('L')  # Convert to grayscale
#             imageNp = np.array(img, 'uint8')
#             id = int(os.path.split(image)[1].split('.')[1])

#             faces.append(imageNp)
#             ids.append(id)

#         ids = np.array(ids)

#         # Train the classifier
#         clf = cv2.face.LBPHFaceRecognizer_create()
#         clf.train(faces, ids)
#         clf.write("classifier.xml")
        
#         # Save training stats and results
#         results_dir = os.path.join(os.getcwd(), "results")
#         if not os.path.exists(results_dir):
#             os.makedirs(results_dir)
        
#         # Save training stats
#         with open(os.path.join(results_dir, "training_stats.txt"), "w") as f:
#             f.write(f"Total images trained: {len(faces)}\n")
#             f.write(f"Unique IDs: {len(set(ids))}\n")
        
#         # Save histogram of training data
#         plt.figure(figsize=(10, 5))
#         plt.hist(ids, bins=len(set(ids)))
#         plt.title("Distribution of Training Images per ID")
#         plt.xlabel("ID")
#         plt.ylabel("Number of Images")
#         plt.savefig(os.path.join(results_dir, "training_distribution.png"))
        
#         print("Training completed!")
#         print(f"Training stats and results saved in {results_dir}")

# def main():
#     trainer = FaceTrainer()
#     trainer.generate_dataset()
#     trainer.train_classifier()

# if __name__ == "__main__":
#     main()


import cv2
import os
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import sqlite3
from datetime import datetime

class FaceTrainer:
    def __init__(self):
        # ... existing code ...
        self.recognizer = cv2.face.LBPHFaceRecognizer_create()
        self.recognizer.read("classifier.xml")
        self.font = cv2.FONT_HERSHEY_SIMPLEX

    # ... existing methods ...

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
                    # Fetch the username from the database
                    conn = sqlite3.connect(self.db_path)
                    cursor = conn.cursor()
                    cursor.execute("SELECT username FROM training_sessions WHERE id=?", (id_,))
                    result = cursor.fetchone()
                    conn.close()

                    if result:
                        username = result[0]
                        cv2.putText(frame, f"{username} ({confidence}%)", (x, y-10), self.font, 0.9, (0, 255, 0), 2)
                    else:
                        cv2.putText(frame, f"Unknown ({confidence}%)", (x, y-10), self.font, 0.9, (0, 255, 0), 2)
                else:
                    cv2.putText(frame, "Unknown", (x, y-10), self.font, 0.9, (0, 0, 255), 2)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def train_classifier(self):
        # ... existing code ...

        # Train the classifier
        self.recognizer.train(faces, np.array(ids))
        self.recognizer.write("classifier.xml")
        
        # ... rest of the existing code ...

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
    