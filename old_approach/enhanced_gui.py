import tkinter as tk
from tkinter import ttk, messagebox, Toplevel
import sqlite3
import traceback
import cv2
import os
import numpy as np
from PIL import Image, ImageTk
import mysql.connector

from old_approach.student import Student
from old_approach.train import Train

class FaceRecognitionSystem:
    def __init__(self, root):
        print("Initializing Face Recognition System")
        self.root = root
        self.root.geometry("800x600")
        self.root.title("Face Recognition Attendance System")

        try:
            # Create SQLite database
            self.create_database()

            # Main frame
            print("Creating main frame")
            main_frame = ttk.Frame(root, padding="20")
            main_frame.pack(fill=tk.BOTH, expand=True)

            # Title
            print("Adding title label")
            title_label = ttk.Label(main_frame, text="Face Recognition Attendance System", font=("Helvetica", 18, "bold"))
            title_label.pack(pady=20)

            # Buttons
            print("Creating buttons")
            button_frame = ttk.Frame(main_frame)
            button_frame.pack(pady=20)

            buttons = [
                ("Student Management", self.open_student_management),
                ("Face Detection", self.face_detection),
                ("Train Model", self.train_model),
                ("View Attendance", self.view_attendance)
            ]

            for text, command in buttons:
                print(f"Adding button: {text}")
                btn = ttk.Button(button_frame, text=text, command=command, width=20)
                btn.pack(pady=10)

            print("Face Recognition System initialized")
        except Exception as e:
            print(f"Error during initialization: {e}")
            print(traceback.format_exc())

    def create_database(self):
        print("Creating database...")
        try:
            conn = sqlite3.connect('face_recognition.db')
            cursor = conn.cursor()
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY,
                    name TEXT NOT NULL,
                    department TEXT
                )
            ''')
            conn.commit()
            conn.close()
            print("Database created successfully")
        except Exception as e:
            print(f"Error creating database: {e}")
            print(traceback.format_exc())

    def open_student_management(self):
        print("Opening Student Management")
        self.new_window = Toplevel(self.root)
        self.app = Student(self.new_window)

    def face_detection(self):
        print("Starting Face Detection")
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        recognizer = cv2.face.LBPHFaceRecognizer_create()
        recognizer.read("classifier.xml")

        cap = cv2.VideoCapture(0)

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.3, 5)

            for (x, y, w, h) in faces:
                cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
                id, confidence = recognizer.predict(gray[y:y+h, x:x+w])

                if confidence < 100:
                    # Here you would typically fetch the name associated with the id from your database
                    name = f"ID: {id}"
                    confidence = f"  {round(100 - confidence)}%"
                else:
                    name = "unknown"
                    confidence = f"  {round(100 - confidence)}%"

                cv2.putText(frame, str(name), (x+5, y-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 255), 2)
                cv2.putText(frame, str(confidence), (x+5, y+h-5), cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 1)

            cv2.imshow('Face Recognition', frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllWindows()

    def train_model(self):
        print("Training Model")
        data_dir = ("data")
        path = [os.path.join(data_dir, file) for file in os.listdir(data_dir)]
        
        faces = []
        ids = []

        for image in path:
            img = Image.open(image).convert('L')
            imageNp = np.array(img, 'uint8')
            id = int(os.path.split(image)[1].split('.')[1])

            faces.append(imageNp)
            ids.append(id)

        ids = np.array(ids)

        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.train(faces, ids)
        clf.write("classifier.xml")

        messagebox.showinfo("Success", "Training datasets completed!")

    def view_attendance(self):
        print("Viewing Attendance")
        try:
            conn = mysql.connector.connect(host="localhost", username="root", password="", database="face_recognition")
            my_cursor = conn.cursor()
            my_cursor.execute("SELECT * FROM attendance")
            data = my_cursor.fetchall()

            if not data:
                messagebox.showinfo("No Data", "No attendance records found.")
                return

            attendance_window = Toplevel(self.root)
            attendance_window.title("Attendance Records")
            attendance_window.geometry("800x600")

            tree = ttk.Treeview(attendance_window)
            tree["columns"] = ("ID", "Name", "Date", "Time")

            tree.column("#0", width=0, stretch=tk.NO)
            tree.column("ID", anchor=tk.W, width=100)
            tree.column("Name", anchor=tk.W, width=200)
            tree.column("Date", anchor=tk.W, width=150)
            tree.column("Time", anchor=tk.W, width=150)

            tree.heading("ID", text="ID", anchor=tk.W)
            tree.heading("Name", text="Name", anchor=tk.W)
            tree.heading("Date", text="Date", anchor=tk.W)
            tree.heading("Time", text="Time", anchor=tk.W)

            for record in data:
                tree.insert("", tk.END, values=record)

            tree.pack(expand=True, fill=tk.BOTH)

            conn.close()

        except Exception as e:
            messagebox.showerror("Error", f"Due To: {str(e)}")

if __name__ == "__main__":
    print("Starting Face Recognition Attendance System")
    try:
        root = tk.Tk()
        app = FaceRecognitionSystem(root)
        print("Entering main event loop")
        root.mainloop()
        print("Face Recognition Attendance System closed")
    except Exception as e:
        print(f"Error in main execution: {e}")
        print(traceback.format_exc())
