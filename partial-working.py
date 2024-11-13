import tkinter as tk
from PIL import Image, ImageTk
from tkinter import Tk, BOTH, X
from tkinter.ttk import Frame, Label, Progressbar
import tkinter.messagebox
import time
import os
import cv2
from ultralytics import YOLO
import numpy as np

# YOLO Model Initialization
model = YOLO('best.pt')  # Load the pre-trained YOLO model

width = 300
height = 200
userId = "admin"
password = "1234"

from tkinter import Label, Entry, Frame, Button, X


class Application(object):
    def __init__(self, master):
        self.master = master
        self.topFrame = Frame(master, height=200, bg="white")
        self.topFrame.pack(fill=X)
        self.bottomFrame = Frame(master, height=600, bg="#f0eec5")
        self.bottomFrame.pack(fill=X)
        self.label1 = Label(self.topFrame, text="TRAFFIC MANAGEMENT SYSTEM", font="BodoniMTBlack 40 bold", bg="white", fg="blue")
        self.label1.place(x=350, y=70)

        self.name = Label(self.bottomFrame, text="NAME: ", font="arial 11 bold", bg="#f0eec5", fg="black")
        self.name.place(x=550, y=210)
        self.password = Label(self.bottomFrame, text="PASSWORD:", font="arial 11 bold", bg="#f0eec5", fg="black")
        self.password.place(x=550, y=240)

        image = Image.open('picture.png')
        image = image.resize((150, 150), Image.LANCZOS)
        img_pro = ImageTk.PhotoImage(image)
        self.label_pic = Label(self.bottomFrame, image=img_pro, background="#f0eec5")
        self.label_pic.image = img_pro
        self.label_pic.place(x=630, y=10)

        self.name1 = tk.StringVar(master)
        self.pwd1 = tk.StringVar(master)
        self.entry_name = Entry(self.bottomFrame, textvariable=self.name1)
        self.entry_name.place(x=650, y=210, width=120, height=20)

        self.entry_pwd = Entry(self.bottomFrame, show="*", width=120, textvariable=self.pwd1)
        self.entry_pwd.place(x=650, y=240, width=120, height=20)

        self.button_register = Button(self.bottomFrame, text="ENTER", font="arial 16 bold ", fg="red", command=self.getThere)
        self.button_register.place(x=660, y=300)

    def getThere(self):
        if self.name1.get() != userId or self.pwd1.get() != password:
            tkinter.messagebox.showinfo("Window Title", "Wrong Username or Password Entered. Can't Proceed Further")
        else:
            there = Traffic()


class newWindow(object):
    def __init__(self, master):
        object.__init__(self)
        w, h = object.winfo_screenwidth(), object.winfo_screenheight()
        object.geometry("%dx%d+0+0" % (w, h))
        object.mainloop()
        w, h = object.winfo_screenwidth(), object.winfo_screenheight()


def main(master):
    app = Application(master)
    master.title("LOGIN PAGE")
    master.geometry("{0}x{1}+0+0".format(root.winfo_screenwidth(), root.winfo_screenheight()))
    master.mainloop()


class Traffic(tk.Toplevel):
    def __init__(self):
        tk.Toplevel.__init__(self)
        self.title("TRAFFIC MANAGEMENT SYSTEM")
        self.configure(background="white")
        self.geometry("{0}x{1}+0+0".format(self.winfo_screenwidth(), self.winfo_screenheight()))
        self.initUI()

    def initUI(self):
        self.image_path = './5.jpg'  # Set the path to your image
        self.vehicle_count = self.predict_vehicle_count(self.image_path)  # Predict the number of vehicles
        self.predicted_time = self.calculate_time(self.vehicle_count)  # Calculate time based on vehicles
        self.display_info()

    def predict_vehicle_count(self, image_path):
        """
        Use YOLO to detect vehicles in the image and return the vehicle count.
        """
        # Load and process the image
        image = cv2.imread(image_path)
        results = model.predict(source=image, imgsz=640, conf=0.5)  # YOLO model prediction

        # Get the count of detected vehicles
        vehicles = results[0].boxes.cls  # Get the class of detected objects (vehicle class id)
        vehicle_count = len(vehicles)  # Number of vehicles detected

        return vehicle_count

    def calculate_time(self, vehicle_count):
        """
        Calculate the predicted traffic time based on the number of detected vehicles.
        """
        # Example time calculation (you can adjust the logic based on your requirements)
        # Assuming each vehicle adds 2 minutes of delay
        predicted_time = vehicle_count * 2  # Just a placeholder logic
        return predicted_time

    def display_info(self):
        """
        Display the vehicle count and predicted traffic time in the window.
        """
        label_vehicles = Label(self, text=f"Vehicles Detected: {self.vehicle_count}", font="Arial 16", bg="white")
        label_vehicles.pack(pady=20)

        label_time = Label(self, text=f"Predicted Traffic Time: {self.predicted_time} minutes", font="Arial 16", bg="white")
        label_time.pack(pady=20)


root = tk.Tk()
main(root)
