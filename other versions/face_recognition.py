from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import messagebox
import mysql.connector
import cv2
import os
import numpy as np

class Face_Recognition:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recognition System")

        title_lbl = Label(self.root, text="FACE RECOGNITION", font=("Times New Roman", 35, "bold"), bg="white", fg="green")
        title_lbl.place(x=0, y=0, width=1530, height=45)

        # 1st image
        img_top = Image.open(r"images\3.jpg")
        img_top = img_top.resize((650, 700), Image.LANCZOS)
        self.photoimg_top = ImageTk.PhotoImage(img_top)
        f_lbl_top = Label(self.root, image=self.photoimg_top)
        f_lbl_top.place(x=0, y=55, width=650, height=700)

        # 2nd image
        img_bottom = Image.open(r"images\3.jpg")
        img_bottom = img_bottom.resize((950, 700), Image.LANCZOS) 
        self.photoimg_bottom = ImageTk.PhotoImage(img_bottom)
        f_lbl_bottom = Label(self.root, image=self.photoimg_bottom)
        f_lbl_bottom.place(x=650, y=55, width=950, height=700)

        # Button
        b1_1 = Button(f_lbl_bottom, text="Face Recognition", command=self.face_recog, cursor="hand2",
                     font=("Times New Roman", 18, "bold"), bg="darkgreen", fg="white")  
        b1_1.place(x=365, y=620, width=200, height=40)

    # ======= face recognition ======
    def face_recog(self):
        def draw_boundary(img, classifier, scaleFactor, minNeighbors, color, text, clf):
            gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            features = classifier.detectMultiScale(gray_image, scaleFactor, minNeighbors)

            coord = []

            for (x, y, w, h) in features:
                cv2.rectangle(img, (x, y), (x + w, y + h), color, 3) 
                id, predict = clf.predict(gray_image[y:y + h, x:x + w])
                confidence = int((100 * (1 - predict / 300)))  

                # Connect to database
                try:
                    conn = mysql.connector.connect(host="localhost", username="root", password="", database="face_recognition")
                    my_cursor = conn.cursor()
                    my_cursor.execute("SELECT Name FROM student WHERE Student_id=%s", (id,))
                    result = my_cursor.fetchone()
                    conn.close()
                except Exception as e:
                    messagebox.showerror("Database Error", f"An error occurred: {str(e)}", parent=self.root)
                    return coord

                if result is not None:
                    name = result[0]
                else:
                    name = "Unknown"

                if confidence > 77:
                    cv2.putText(img, f"Name: {name}", (x, y - 55), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)
                else:
                    cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 3)  
                    cv2.putText(img, "Unknown Face", (x, y - 55), cv2.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 3)

                coord = [x, y, w, h]

            return coord

        def recognize(img, clf, faceCascade):
            coord = draw_boundary(img, faceCascade, 1.1, 10, (255, 255, 255), "Face", clf)
            return img

        faceCascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
        clf = cv2.face.LBPHFaceRecognizer_create()
        clf.read("classifier.xml")

        video_cap = cv2.VideoCapture(0)
        while True:
            ret, img = video_cap.read()
            if not ret:
                messagebox.showerror("Error", "Failed to capture video from camera", parent=self.root)
                break
            img = recognize(img, clf, faceCascade)
            cv2.imshow("Welcome to face recognition", img)

            if cv2.waitKey(1) == 13:  # Enter key to exit
                break

        video_cap.release()
        cv2.destroyAllWindows()
