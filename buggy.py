import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from typing import List, Tuple, Dict
import logging
from dataclasses import dataclass
import threading
import queue
from ultralytics import YOLO
import torch

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    """Data class to store detection results for each image"""
    image_path: str
    vehicle_count: int
    predicted_time: float
    detection_boxes: np.ndarray
    confidence_scores: List[float]

class VehicleDetector:
    """Class to handle YOLOv8 model operations and vehicle detection"""
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        """
        Initialize the YOLOv8 model
        
        Args:
            model_path: Path to pre-trained YOLOv8 model
            conf_threshold: Confidence threshold for detections
        """
        try:
            self.model = YOLO(model_path)
            self.conf_threshold = conf_threshold
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            logger.info(f"Successfully loaded YOLOv8 model on {self.device}")
            
            # Define vehicle classes based on COCO dataset
            self.vehicle_classes = {
                2: 'car',
                3: 'motorcycle',
                5: 'bus',
                7: 'truck',
                # Add more vehicle classes as needed
            }
            
        except Exception as e:
            logger.error(f"Failed to load YOLOv8 model: {e}")
            raise

    def predict(self, image: np.ndarray) -> Tuple[int, np.ndarray, List[float]]:
        """
        Detect vehicles in an image
        
        Args:
            image: Input image as numpy array
            
        Returns:
            Tuple containing vehicle count, detection boxes, and confidence scores
        """
        try:
            # Run inference
            results = self.model(image, conf=self.conf_threshold)[0]
            
            # Get detection boxes and scores
            boxes = []
            scores = []
            vehicle_count = 0
            
            # Process each detection
            for box in results.boxes:
                # Get class ID and confidence score
                cls = int(box.cls[0].item())  # Convert tensor to integer
                conf = float(box.conf[0].item())  # Convert tensor to float
                
                # Check if detected object is a vehicle
                if cls in self.vehicle_classes:
                    vehicle_count += 1
                    # Convert box coordinates to x, y, w, h format
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    w = x2 - x1
                    h = y2 - y1
                    boxes.append([x1, y1, w, h])
                    scores.append(conf)
                    
                    # Log detection for debugging
                    logger.debug(f"Detected {self.vehicle_classes[cls]} with confidence {conf:.2f}")
            
            logger.info(f"Total vehicles detected: {vehicle_count}")
            return vehicle_count, np.array(boxes), scores
            
        except Exception as e:
            logger.error(f"Error during prediction: {e}")
            return 0, np.array([]), []

class SignalTimingPredictor:
    """Class to predict green signal timing based on vehicle count"""
    @staticmethod
    def predict_time(vehicle_count: int) -> float:
        """
        Predict green signal time based on vehicle count
        
        Args:
            vehicle_count: Number of vehicles detected
            
        Returns:
            Predicted time in seconds
        """
        base_time = 30  # minimum green time
        time_per_vehicle = 3  # additional seconds per vehicle
        return min(base_time + (vehicle_count * time_per_vehicle), 120)  # max 120 seconds

class TrafficAnalysisGUI:
    """Main GUI class for traffic analysis application"""
    def __init__(self, root: tk.Tk, image_folder: str, model_path: str):
        """
        Initialize the GUI
        
        Args:
            root: Tkinter root window
            image_folder: Path to folder containing images
            model_path: Path to pre-trained YOLOv8 model
        """
        self.root = root
        self.root.title("Traffic Analysis System")
        
        # Initialize components
        self.detector = VehicleDetector(model_path)
        self.predictor = SignalTimingPredictor()
        self.image_paths = self._get_image_paths(image_folder)
        self.results_queue = queue.Queue()
        
        # Setup GUI layout
        self.setup_gui()
        
        # Start processing thread
        self.processing_thread = threading.Thread(target=self._process_images)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        # Start update loop
        self.root.after(100, self._update_gui)

    def setup_gui(self):
        """Setup the GUI layout"""
        # Create main frame with grid 3x3
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(padx=10, pady=10)
        
        # Create image panels and progress bars
        self.panels = []
        self.progress_vars = []
        self.count_labels = []
        self.time_labels = []
        
        style = ttk.Style()
        style.configure("Green.Horizontal.TProgressbar", background='green')
        
        for i in range(3):
            for j in range(3):
                frame = ttk.Frame(self.main_frame)
                frame.grid(row=i, column=j, padx=5, pady=5)
                
                # Image panel
                panel = ttk.Label(frame)
                panel.pack()
                self.panels.append(panel)
                
                # Vehicle count label
                count_label = ttk.Label(frame, text="Vehicles: --")
                count_label.pack()
                self.count_labels.append(count_label)
                
                # Time prediction label
                time_label = ttk.Label(frame, text="Time: --")
                time_label.pack()
                self.time_labels.append(time_label)
                
                # Progress bar
                progress_var = tk.DoubleVar()
                progress_bar = ttk.Progressbar(
                    frame, 
                    variable=progress_var,
                    maximum=100,
                    length=200,
                    mode='determinate',
                    style="Green.Horizontal.TProgressbar"
                )
                progress_bar.pack()
                self.progress_vars.append(progress_var)

    def _get_image_paths(self, folder: str) -> List[str]:
        """Get paths of all images in the specified folder"""
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        image_paths = [
            os.path.join(folder, f) for f in os.listdir(folder)
            if os.path.splitext(f)[1].lower() in valid_extensions
        ]
        # Sort to ensure consistent order
        image_paths.sort()
        return image_paths[:9]  # Limit to 9 images

    def _process_images(self):
        """Process images in a separate thread"""
        for idx, image_path in enumerate(self.image_paths):
            try:
                # Read and process image
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Failed to read image: {image_path}")
                
                # Detect vehicles
                vehicle_count, boxes, scores = self.detector.predict(image)
                
                # Predict signal timing
                predicted_time = self.predictor.predict_time(vehicle_count)
                
                # Put results in queue
                self.results_queue.put((idx, DetectionResult(
                    image_path=image_path,
                    vehicle_count=vehicle_count,
                    predicted_time=predicted_time,
                    detection_boxes=boxes,
                    confidence_scores=scores
                )))
                
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
                continue

    def _update_gui(self):
        """Update GUI with processed results"""
        try:
            while True:
                # Get results from queue
                idx, result = self.results_queue.get_nowait()
                
                # Update image with detections
                image = cv2.imread(result.image_path)
                for box, score in zip(result.detection_boxes, result.confidence_scores):
                    x, y, w, h = map(int, box)
                    cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
                    cv2.putText(image, f'{score:.2f}', (x, y - 5),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                # Convert to PhotoImage
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image)
                image = image.resize((300, 225))  # Larger size for better visibility
                photo = ImageTk.PhotoImage(image)
                
                # Update GUI elements
                self.panels[idx].configure(image=photo)
                self.panels[idx].image = photo  # Keep reference
                
                self.count_labels[idx].configure(
                    text=f"Vehicles: {result.vehicle_count}"
                )
                self.time_labels[idx].configure(
                    text=f"Time: {result.predicted_time:.1f}s"
                )
                
                # Update progress bar
                self.progress_vars[idx].set(
                    (result.predicted_time / 120) * 100  # 120s is max time
                )
                
        except queue.Empty:
            pass
        
        # Schedule next update
        self.root.after(100, self._update_gui)

def main():
    """Main function to run the application"""
    # Setup command line arguments
    import argparse
    parser = argparse.ArgumentParser(description='Traffic Analysis System')
    parser.add_argument('--image_folder', required=True, help='Path to folder containing images')
    parser.add_argument('--model_path', required=True, help='Path to pre-trained YOLOv8 model')
    args = parser.parse_args()

    # Create and run GUI
    root = tk.Tk()
    app = TrafficAnalysisGUI(root, args.image_folder, args.model_path)
    root.mainloop()

if __name__ == "__main__":
    main()