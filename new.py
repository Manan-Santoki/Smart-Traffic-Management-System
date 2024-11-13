import tkinter as tk
from tkinter import ttk
from tkinter import Label, PhotoImage
from PIL import Image, ImageTk
import cv2
import torch  # YOLO model is assumed to be a PyTorch model
import time

# Load the pre-trained YOLO model (YOLOv5 here as an example)
model = torch.hub.load('ultralytics/yolov5', 'yolov5s', pretrained=True)
model.conf = 0.5  # Set confidence threshold for detections

# Configuration
image_folder = "./5"
image_files = ["1.jpg", "2.jpg", "3.jpg", "4.jpg", "5.jpg", "6.jpg", "7.jpg", "8.jpg", "9.jpg"]

# Define detection function
def detect_vehicles(image_path):
    # Load image and run detection
    img = Image.open(image_path)
    results = model(img)
    
    # Filter results for vehicles (e.g., car, truck, bus)
    vehicles = results.pandas().xyxy[0]
    vehicle_count = vehicles[vehicles['name'].isin(['car', 'truck', 'bus'])].shape[0]
    
    # Estimate green light time based on number of vehicles
    green_light_time = vehicle_count * 2  # 2 seconds per vehicle as an example
    return vehicle_count, green_light_time, results.ims[0]

# Initialize GUI
root = tk.Tk()
root.title("Vehicle Detection and Green Light Prediction")
root.geometry("1200x800")

# Prepare frames to hold each image and its details
frames = []
vehicle_counts = []
green_light_times = []
images = []

# Load and display images
for idx, img_file in enumerate(image_files):
    # Detect vehicles
    vehicle_count, green_light_time, img = detect_vehicles(f"{image_folder}/{img_file}")
    vehicle_counts.append(vehicle_count)
    green_light_times.append(green_light_time)
    
    # Convert to Tkinter image
    img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    img.thumbnail((200, 200))  # Resize for display
    tk_img = ImageTk.PhotoImage(img)
    images.append(tk_img)  # Store reference to avoid garbage collection
    
    # Create a frame for each image
    frame = tk.Frame(root, relief=tk.RAISED, borderwidth=1)
    frame.grid(row=idx // 3, column=idx % 3, padx=10, pady=10)
    frames.append(frame)
    
    # Display image
    img_label = Label(frame, image=tk_img)
    img_label.pack()
    
    # Display vehicle count
    count_label = Label(frame, text=f"Vehicles: {vehicle_count}")
    count_label.pack()
    
    # Display green light time
    time_label = Label(frame, text=f"Green Light Time: {green_light_time}s")
    time_label.pack()
    
    # Progress bar
    progress = ttk.Progressbar(frame, length=150, maximum=green_light_time)
    progress.pack(pady=5)
    progress['value'] = green_light_time  # Set progress to full based on green light time

# Run the Tkinter GUI
root.mainloop()
