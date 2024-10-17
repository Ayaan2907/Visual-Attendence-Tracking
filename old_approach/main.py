from tkinter import *
from tkinter import ttk
from PIL import Image, ImageTk

from old_approach.student import Student


class Face_Recognition_System:
    def __init__(self, root):
        self.root = root
        self.root.geometry("1530x790+0+0")
        self.root.title("Face Recognition System")

        # first image
        img1 = Image.open(r"/images/2.jpeg")
        img1= img1.resize((500,130), Image.LANCZOS)
        self.photoimg1=ImageTk.PhotoImage(img1)
        f_lbl = Label(self.root, image=self.photoimg1)
        f_lbl.place(x=0, y=0, width=500, height=130)

        # second image
        img2 = Image.open(r"/images/1.jpg")
        img2= img2.resize((500,130), Image.LANCZOS)
        self.photoimg2=ImageTk.PhotoImage(img2)
        f_lbl = Label(self.root, image=self.photoimg2)
        f_lbl.place(x=500, y=0, width=500, height=130)

         # Third image
        img3 = Image.open(r"/images/3.jpg")
        img3= img3.resize((560,130), Image.LANCZOS)
        self.photoimg3=ImageTk.PhotoImage(img3)
        f_lbl = Label(self.root, image=self.photoimg3)
        f_lbl.place(x=1000, y=0, width=560, height=130)

         # bg image
        img4 = Image.open(r"/images/3.jpg")
        img4= img4.resize((1530,710), Image.LANCZOS)
        self.photoimg4=ImageTk.PhotoImage(img4)

        bg_img = Label(self.root, image=self.photoimg4)
        bg_img.place(x=0, y=130, width=1530, height=710)

        title_lbl = Label(bg_img, text = "FACE RECOGNITION ATTENDANCE SYSTEM", font=("Times New Roman", 35, "bold"), bg="White", fg="red")
        title_lbl.place(x=0, y=0,width=1530, height=45)

        # Student button
        img5 = Image.open(r"/images/3.jpg")
        img5= img5.resize((220,220), Image.LANCZOS)
        self.photoimg5=ImageTk.PhotoImage(img5)
        b1= Button(bg_img, image=self.photoimg5, command=self.student_details,cursor="hand2")
        b1.place(x=200, y=100, width=220, height=220)

        b1_1= Button(bg_img, text="Student Details", command=self.student_details ,cursor="hand2" , font=("Times New Roman", 15, "bold"), bg="darkblue", fg="white")
        b1_1.place(x=200, y=300, width=220, height=40)

         # Detect face button
        img6 = Image.open(r"/images/3.jpg")
        img6= img6.resize((220,220), Image.LANCZOS)
        self.photoimg6=ImageTk.PhotoImage(img6)
        b1= Button(bg_img, image=self.photoimg6,cursor="hand2")
        b1.place(x=500, y=100, width=220, height=220)

        b1_1= Button(bg_img, text="Face Detector",cursor="hand2" , font=("Times New Roman", 15, "bold"), bg="darkblue", fg="white")
        b1_1.place(x=500, y=300, width=220, height=40)

         # Attendance button
        img7 = Image.open(r"/images/3.jpg")
        img7= img7.resize((220,220), Image.LANCZOS)
        self.photoimg7=ImageTk.PhotoImage(img7)
        b1= Button(bg_img, image=self.photoimg7,cursor="hand2")
        b1.place(x=800, y=100, width=220, height=220)

        b1_1= Button(bg_img, text="Attendance",cursor="hand2" , font=("Times New Roman", 15, "bold"), bg="darkblue", fg="white")
        b1_1.place(x=800, y=300, width=220, height=40)

         # Train face button
        img8 = Image.open(r"/images/3.jpg")
        img8= img8.resize((220,220), Image.LANCZOS)
        self.photoimg8=ImageTk.PhotoImage(img8)
        b1= Button(bg_img, image=self.photoimg8,cursor="hand2")
        b1.place(x=200, y=380, width=220, height=220)

        b1_1= Button(bg_img, text="Train Data",cursor="hand2" , font=("Times New Roman", 15, "bold"), bg="darkblue", fg="white")
        b1_1.place(x=200, y=580, width=220, height=40)



            #==============================Function buttons

    def student_details(self):
        self.new_window = Toplevel(self.root)
        self.app= Student(self.new_window)

if __name__ == "__main__":
    root = Tk()
    obj = Face_Recognition_System(root)
    root.mainloop()
