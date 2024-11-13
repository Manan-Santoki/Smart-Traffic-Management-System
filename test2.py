import tkinter as tk
from tkinter import ttk
from PIL import Image, ImageTk
import cv2
import numpy as np
import os
from typing import List, Tuple
import logging
from dataclasses import dataclass
import threading
import queue
from ultralytics import YOLO
import torch

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class DetectionResult:
    image_path: str
    vehicle_count: int
    predicted_time: float
    detection_boxes: np.ndarray
    confidence_scores: List[float]
    class_ids: List[int]

class VehicleDetector:
    def __init__(self, model_path: str, conf_threshold: float = 0.25):
        self.model = YOLO(model_path)
        self.conf_threshold = conf_threshold
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        logger.info(f"YOLO model loaded on {self.device}")
        self.vehicle_classes = {2: 'car', 3: 'motorcycle', 5: 'bus', 7: 'truck'}

    def predict(self, image: np.ndarray) -> Tuple[int, np.ndarray, List[float], List[int]]:
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.model.predict(source=image, conf=self.conf_threshold)
        
        boxes, scores, class_ids = [], [], []
        vehicle_count = 0
        
        for result in results:
            for box in result.boxes:
                cls = int(box.cls.item())
                conf = float(box.conf.item())
                
                if cls in self.vehicle_classes:
                    vehicle_count += 1
                    x1, y1, x2, y2 = box.xyxy.cpu().numpy()
                    w, h = x2 - x1, y2 - y1
                    boxes.append([x1, y1, w, h])
                    scores.append(conf)
                    class_ids.append(cls)
                    logger.debug(f"Detected {self.vehicle_classes[cls]} with confidence {conf:.2f}")
        
        logger.info(f"Total vehicles detected: {vehicle_count}")
        return vehicle_count, np.array(boxes), scores, class_ids

class SignalTimingPredictor:
    @staticmethod
    def predict_time(vehicle_count: int) -> float:
        base_time, time_per_vehicle, max_time = 30, 3, 120
        return min(base_time + (vehicle_count * time_per_vehicle), max_time)

class TrafficAnalysisGUI:
    def __init__(self, root: tk.Tk, image_folder: str, model_path: str):
        self.root = root
        self.root.title("Traffic Analysis System")
        self.detector = VehicleDetector(model_path)
        self.predictor = SignalTimingPredictor()
        self.image_paths = self._get_image_paths(image_folder)
        self.results_queue = queue.Queue()
        
        self.setup_gui()
        threading.Thread(target=self._process_images, daemon=True).start()
        self.root.after(100, self._update_gui)

    def setup_gui(self):
        self.main_frame = ttk.Frame(self.root)
        self.main_frame.pack(padx=10, pady=10)
        
        self.panels, self.progress_vars, self.count_labels, self.time_labels = [], [], [], []
        style = ttk.Style()
        style.configure("Green.Horizontal.TProgressbar", background='green')
        
        for i in range(3):
            for j in range(3):
                frame = ttk.Frame(self.main_frame)
                frame.grid(row=i, column=j, padx=5, pady=5)
                panel = ttk.Label(frame)
                panel.pack()
                self.panels.append(panel)
                
                count_label = ttk.Label(frame, text="Vehicles: --")
                count_label.pack()
                self.count_labels.append(count_label)
                
                time_label = ttk.Label(frame, text="Time: --")
                time_label.pack()
                self.time_labels.append(time_label)
                
                progress_var = tk.DoubleVar()
                progress_bar = ttk.Progressbar(frame, variable=progress_var, maximum=100, length=200, mode='determinate', style="Green.Horizontal.TProgressbar")
                progress_bar.pack()
                self.progress_vars.append(progress_var)

    def _get_image_paths(self, folder: str) -> List[str]:
        valid_extensions = {'.jpg', '.jpeg', '.png'}
        image_paths = [os.path.join(folder, f) for f in os.listdir(folder) if os.path.splitext(f)[1].lower() in valid_extensions]
        image_paths.sort()
        return image_paths[:9]

    def _process_images(self):
        for idx, image_path in enumerate(self.image_paths):
            try:
                image = cv2.imread(image_path)
                if image is None:
                    raise ValueError(f"Failed to read image: {image_path}")
                
                vehicle_count, boxes, scores, class_ids = self.detector.predict(image)
                predicted_time = self.predictor.predict_time(vehicle_count)
                
                self.results_queue.put((idx, DetectionResult(image_path=image_path, vehicle_count=vehicle_count, predicted_time=predicted_time, detection_boxes=boxes, confidence_scores=scores, class_ids=class_ids)))
                
            except Exception as e:
                logger.error(f"Error processing image {image_path}: {e}")
                continue

    def _update_gui(self):
        try:
            while True:
                idx, result = self.results_queue.get_nowait()
                image = cv2.imread(result.image_path)
                colors = {2: (0, 255, 0), 3: (255, 0, 0), 5: (0, 0, 255), 7: (255, 255, 0)}
                
                for box, score, cls_id in zip(result.detection_boxes, result.confidence_scores, result.class_ids):
                    x, y, w, h = map(int, box)
                    color = colors.get(cls_id, (0, 255, 0))
                    cv2.rectangle(image, (x, y), (x + w, y + h), color, 2)
                    
                    label = f'{self.detector.vehicle_classes[cls_id]}: {score:.2f}'
                    label_size, _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)
                    label_y = max(y - 5, label_size[1])
                    cv2.rectangle(image, (x, label_y - label_size[1]), (x + label_size[0], label_y + 5), color, -1)
                    cv2.putText(image, label, (x, label_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 2)
                
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(image).resize((300, 225))
                photo = ImageTk.PhotoImage(image)
                
                self.panels[idx].configure(image=photo)
                self.panels[idx].image = photo
                self.count_labels[idx].configure(text=f"Vehicles: {result.vehicle_count}")
                self.time_labels[idx].configure(text=f"Time: {result.predicted_time:.1f}s")
                self.progress_vars[idx].set((result.predicted_time / 120) * 100)
                
        except queue.Empty:
            pass
        
        self.root.after(100, self._update_gui)

def main():
    import argparse
    parser = argparse.ArgumentParser(description='Traffic Analysis System')
    parser.add_argument('--image_folder', required=True, help='Path to folder containing images')
    parser.add_argument('--model_path', required=True, help='Path to YOLOv8 model')
    args = parser.parse_args()

    root = tk.Tk()
    app = TrafficAnalysisGUI(root, args.image_folder, args.model_path)
    root.mainloop()

if __name__ == "__main__":
    main()
