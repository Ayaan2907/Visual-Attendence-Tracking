from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk
from tkinter import messagebox
import mysql.connector
import cv2
import os

class Student:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recognition System")

        # =====variables=====
        self.var_std_id = StringVar()
        self.var_std_name = StringVar()

        # Load pre-defined data on face frontals from OpenCV (Haar Cascade)
        self.cascade_path = "haarcascade_frontalface_default.xml"
        if not os.path.exists(self.cascade_path):
            messagebox.showerror("Error", f"Cannot find Haar cascade file at {self.cascade_path}. Please ensure it is in the correct location.")
            return
        self.face_classifier = cv2.CascadeClassifier(self.cascade_path)

        # First image
        img1 = Image.open(r"/images/2.jpeg")
        img1 = img1.resize((500, 130), Image.LANCZOS)
        self.photoimg1 = ImageTk.PhotoImage(img1)
        f_lbl = Label(self.root, image=self.photoimg1)
        f_lbl.place(x=0, y=0, width=500, height=130)

        # Second image
        img2 = Image.open(r"/images/1.jpg")
        img2 = img2.resize((500, 130), Image.LANCZOS)
        self.photoimg2 = ImageTk.PhotoImage(img2)
        f_lbl = Label(self.root, image=self.photoimg2)
        f_lbl.place(x=500, y=0, width=500, height=130)

        # Third image
        img3 = Image.open(r"/images/3.jpg")
        img3 = img3.resize((560, 130), Image.LANCZOS)
        self.photoimg3 = ImageTk.PhotoImage(img3)
        f_lbl = Label(self.root, image=self.photoimg3)
        f_lbl.place(x=1000, y=0, width=560, height=130)

        # Background image
        img4 = Image.open(r"/images/3.jpg")
        img4 = img4.resize((1530, 710), Image.LANCZOS)
        self.photoimg4 = ImageTk.PhotoImage(img4)

        bg_img = Label(self.root, image=self.photoimg4)
        bg_img.place(x=0, y=130, width=1530, height=710)

        title_lbl = Label(bg_img, text="STUDENT MANAGEMENT SYSTEM", font=("Times New Roman", 35, "bold"), bg="white", fg="darkgreen")
        title_lbl.place(x=0, y=0, width=1530, height=45)

        main_frame = Frame(bg_img, bd=2)
        main_frame.place(x=20, y=50, width=1480, height=600)

        # Left label frame
        Left_Frame = LabelFrame(main_frame, bd=2, bg="white", relief=RIDGE, text="Student Details", font=("times new roman", 12, "bold"))
        Left_Frame.place(x=10, y=10, width=730, height=580)

        img_left = Image.open(r"/images/3.jpg")
        img_left = img_left.resize((720, 130), Image.LANCZOS)
        self.photoimg_left = ImageTk.PhotoImage(img_left)
        f_lbl = Label(Left_Frame, image=self.photoimg_left)
        f_lbl.place(x=5, y=0, width=720, height=130)

        # Current course
        current_course_frame = LabelFrame(Left_Frame, bd=2, bg="white", relief=RIDGE, text="Current Course Information", font=("times new roman", 13, "bold"))
        current_course_frame.place(x=5, y=135, width=720, height=150)

        # Department
        dep_label = Label(current_course_frame, text="Department", font=("times new roman", 13, "bold"), bg="white")
        dep_label.grid(row=0, column=0, padx=10, sticky=W)
        dep_combo = ttk.Combobox(current_course_frame, font=("times new roman", 13, "bold"), state="readonly", width=20)
        dep_combo["values"] = ("Select Department", "Computer", "IT", "Civil", "Mechanical")
        dep_combo.current(0)
        dep_combo.grid(row=0, column=1, padx=2, pady=10, sticky=W)

        # Class Student Frame
        class_Student_Frame = LabelFrame(Left_Frame, bd=2, bg="white", relief=RIDGE, text="Class Student Information", font=("times new roman", 12, "bold"))
        class_Student_Frame.place(x=5, y=250, width=720, height=300)

        # Student ID
        studentId_label = Label(class_Student_Frame, text="Student ID:", font=("times new roman", 13, "bold"), bg="white")
        studentId_label.grid(row=0, column=0, padx=10, sticky=W)
        studentId_entry = ttk.Entry(class_Student_Frame, textvariable=self.var_std_id, width=20, font=("times new roman", 13, "bold"))
        studentId_entry.grid(row=0, column=1, padx=10, sticky=W)

        # Student Name
        studentName_label = Label(class_Student_Frame, text="Student Name:", font=("times new roman", 13, "bold"), bg="white")
        studentName_label.grid(row=0, column=2, padx=10, sticky=W)
        studentName_entry = ttk.Entry(class_Student_Frame, textvariable=self.var_std_name, width=20, font=("times new roman", 13, "bold"))
        studentName_entry.grid(row=0, column=3, padx=10, sticky=W)

        # Buttons Frame
        btn_frame = Frame(class_Student_Frame, bd=2, relief=RIDGE, bg="white")
        btn_frame.place(x=0, y=200, width=715, height=35)

        save_btn = Button(btn_frame, text="Save", command=self.add_data, font=("times new roman", 13, "bold"), bg="blue", fg="white", width=17)
        save_btn.grid(row=0, column=0)

        take_photo_btn = Button(btn_frame, command=self.generate_dataset, text="Take Photo Sample", font=("times new roman", 13, "bold"), bg="blue", fg="white", width=35)
        take_photo_btn.grid(row=0, column=1)

    def add_data(self):
        if self.var_std_name.get() == "" or self.var_std_id.get() == "":
            messagebox.showerror("Error", "All Fields are required", parent=self.root)
        else:
            try:
                conn = mysql.connector.connect(host="localhost", username="root", password="", database="face_recognition")
                my_cursor = conn.cursor()
                my_cursor.execute("INSERT INTO student VALUES(%s, %s)", (self.var_std_id.get(), self.var_std_name.get()))
                conn.commit()
                conn.close()
                messagebox.showinfo("Success", "Student details have been added successfully", parent=self.root)
            except Exception as es:
                messagebox.showerror("Error", f"Due To: {str(es)}", parent=self.root)

    def face_cropped(self, img):
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_classifier.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

        if len(faces) == 0:
            print("No faces detected.")
            return None

        for (x, y, w, h) in faces:
            face_cropped = img[y:y+h, x:x+w]
            print("Face detected and cropped.")
            return face_cropped
        return None

    def generate_dataset(self):
        if self.var_std_name.get() == "" or self.var_std_id.get() == "":
            messagebox.showerror("Error", "All Fields are required", parent=self.root)
        else:
            # Ensure the data directory exists
            save_path = r"/data"
            if not os.path.exists(save_path):
                os.makedirs(save_path)

            cap = cv2.VideoCapture(0)
            img_id = 0
            while True:
                ret, my_frame = cap.read()
                if not ret:
                    print("Failed to capture frame from camera.")
                    break

                face = self.face_cropped(my_frame)
                if face is not None:
                    img_id += 1
                    face = cv2.resize(face, (450, 450))
                    face = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY)
                    file_name_path = os.path.join(save_path, f"user.{self.var_std_id.get()}.{img_id}.jpg")
                    cv2.imwrite(file_name_path, face)
                    cv2.putText(face, str(img_id), (50, 50), cv2.FONT_HERSHEY_COMPLEX, 2, (0, 255, 0), 2)
                    cv2.imshow("Cropped Face", face)
                    print(f"Image {img_id} saved at {file_name_path}.")

                if img_id == 100:  #  capture 100 images to exit
                    break

            cap.release()
            cv2.destroyAllWindows()
            messagebox.showinfo("Result", "Generating datasets completed!", parent=self.root)

if __name__ == "__main__":
    root = Tk()
    obj = Student(root)
    root.mainloop()
