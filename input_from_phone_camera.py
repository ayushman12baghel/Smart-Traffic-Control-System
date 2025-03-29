import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import time
import os
import requests

class PhoneCameraCapture:
    def __init__(self, ip_address='http://192.168.1.100:4747'):
        """
        Initialize phone camera capture using IP Webcam
        
        Args:
            ip_address (str): IP address of your phone's camera streaming app
        """
        self.ip_address = ip_address
        self.stream_url = f'{ip_address}/video'
        self.capture = None

    def connect_camera(self):
        """
        Connect to phone camera via IP Webcam
        
        Returns:
            bool: True if connection successful, False otherwise
        """
        try:
            # Test connection
            response = requests.get(self.ip_address, timeout=5)
            if response.status_code == 200:
                self.capture = cv2.VideoCapture(self.stream_url)
                
                # Optional: Set camera resolution
                self.capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
                self.capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
                
                return True
            return False
        except Exception as e:
            print(f"Connection error: {e}")
            return False

    def capture_image(self, output_path='phone_camera_input.jpg'):
        """
        Capture an image from the phone camera
        
        Args:
            output_path (str): Path to save the captured image
        
        Returns:
            bool: True if image captured successfully, False otherwise
        """
        if not self.capture or not self.capture.isOpened():
            print("Camera not connected. Use connect_camera() first.")
            return False
        
        try:
            # Capture multiple frames to ensure a fresh image
            for _ in range(5):
                ret, frame = self.capture.read()
            
            if not ret:
                print("Failed to capture image")
                return False
            
            # Ensure directory exists
            os.makedirs(os.path.dirname(output_path) or '.', exist_ok=True)
            
            # Save the image
            cv2.imwrite(output_path, frame)
            print(f"Image saved to {output_path}")
            return True
        
        except Exception as e:
            print(f"Error capturing image: {e}")
            return False

    def __del__(self):
        """
        Release camera resources
        """
        if self.capture:
            self.capture.release()

class VehicleDetectionSystem:
    def __init__(self, confidence_threshold=0.5):
        """
        Initialize vehicle detection system using Faster R-CNN
        
        Args:
            confidence_threshold (float): Minimum confidence to consider a detection valid
        """
        # Pre-trained model with latest weights
        weights = FasterRCNN_ResNet50_FPN_Weights.DEFAULT
        self.model = fasterrcnn_resnet50_fpn(weights=weights)
        self.model.eval()
        
        # Get the class names from the weights
        self.class_names = weights.meta["categories"]
        
        # COCO class indices for our specific vehicles
        self.vehicle_indices = {
            'car': self.class_names.index('car'),
            'truck': self.class_names.index('truck'),
            'bus': self.class_names.index('bus'),
            'bicycle': self.class_names.index('bicycle'),
            'motorcycle': self.class_names.index('motorcycle')
        }
        
        # Confidence threshold
        self.confidence_threshold = confidence_threshold
        
    def detect_vehicles(self, image_path):
        """
        Detect vehicles in an image
        
        Args:
            image_path (str): Path to the input image
        
        Returns:
            dict: Counts of different vehicle types
        """
        # Read image
        image = Image.open(image_path)
        
        # Convert to tensor
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        input_image = transform(image).unsqueeze(0)
        
        # Perform inference
        with torch.no_grad():
            prediction = self.model(input_image)
        
        # Process predictions
        vehicle_counts = {
            'cars': 0,
            'trucks': 0,
            'buses': 0,
            'bikes': 0,
            'rickshaws': 0  # Note: No direct rickshaw detection
        }
        
        # Get detections above confidence threshold
        boxes = prediction[0]['boxes']
        labels = prediction[0]['labels']
        scores = prediction[0]['scores']
        
        for box, label, score in zip(boxes, labels, scores):
            if score > self.confidence_threshold:
                try:
                    # Find the class name for this label
                    detected_class = self.class_names[label.item()]
                    
                    # Map detections to our model's categories
                    if detected_class == 'car':
                        vehicle_counts['cars'] += 1
                    elif detected_class == 'truck':
                        vehicle_counts['trucks'] += 1
                    elif detected_class == 'bus':
                        vehicle_counts['buses'] += 1
                    elif detected_class in ['bicycle', 'motorcycle']:
                        vehicle_counts['bikes'] += 1
                except Exception as e:
                    print(f"Error processing detection: {e}")
        
        return vehicle_counts
    
    def visualize_detections(self, image_path, output_path='detected_vehicles.jpg'):
        """
        Visualize vehicle detections on the image
        
        Args:
            image_path (str): Path to input image
            output_path (str): Path to save output image
        """
        # Read image
        image = Image.open(image_path)
        transform = torchvision.transforms.Compose([
            torchvision.transforms.ToTensor()
        ])
        input_image = transform(image).unsqueeze(0)
        
        # Perform inference
        with torch.no_grad():
            prediction = self.model(input_image)
        
        # Prepare visualization
        original_image = cv2.imread(image_path)
        original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(15,10))
        plt.imshow(original_image)
        
        # Draw bounding boxes
        for box, label, score in zip(
            prediction[0]['boxes'], 
            prediction[0]['labels'], 
            prediction[0]['scores']
        ):
            if score > self.confidence_threshold:
                # Convert box coordinates
                box = box.numpy()
                plt.gca().add_patch(
                    plt.Rectangle(
                        (box[0], box[1]), 
                        box[2] - box[0], 
                        box[3] - box[1], 
                        fill=False, 
                        edgecolor='red', 
                        linewidth=2
                    )
                )
                
                # Add label
                try:
                    class_name = self.class_names[label.item()]
                    plt.text(
                        box[0], box[1] - 10, 
                        f'{class_name} {score:.2f}', 
                        color='red'
                    )
                except Exception as e:
                    print(f"Error adding label: {e}")
        
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(output_path)
        plt.close()

class TrafficSignalOptimizer:
    def __init__(self, model_path='models/timing_model.pkl', scaler_path='models/timing_scaler.pkl'):
        """
        Initialize traffic signal optimizer using existing model
        
        Args:
            model_path (str): Path to trained RandomForest model
            scaler_path (str): Path to feature scaler
        """
        import pickle
        
        # Load trained model and scaler
        try:
            with open(model_path, 'rb') as f:
                self.model = pickle.load(f)
            
            with open(scaler_path, 'rb') as f:
                self.scaler = pickle.load(f)
        except FileNotFoundError:
            print("Warning: Model or scaler files not found. Using default values.")
            self.model = None
            self.scaler = None
    
    def predict_signal_timing(self, vehicle_counts, hour=None, day_of_week=None):
        """
        Predict optimal signal timing based on vehicle counts
        
        Args:
            vehicle_counts (dict): Counts of different vehicle types
            hour (int, optional): Hour of the day
            day_of_week (int, optional): Day of the week
        
        Returns:
            int: Recommended green signal time in seconds
        """
        # Use current time if not provided
        if hour is None:
            hour = int(time.strftime('%H'))
        if day_of_week is None:
            day_of_week = int(time.strftime('%w'))
        
        # If no model is loaded, use a simple heuristic
        if self.model is None or self.scaler is None:
            return self._simple_timing_heuristic(vehicle_counts)
        
        # Prepare input data
        input_data = pd.DataFrame({
            'cars': [vehicle_counts['cars']],
            'bikes': [vehicle_counts['bikes']],
            'buses': [vehicle_counts['buses']],
            'trucks': [vehicle_counts['trucks']],
            'rickshaws': [vehicle_counts.get('rickshaws', 0)],
            'hour': [hour],
            'day_of_week': [day_of_week]
        })
        
        # Scale features
        input_scaled = self.scaler.transform(input_data)
        
        # Predict green time
        green_time = self.model.predict(input_scaled)[0]
        
        # Ensure prediction is within bounds
        green_time = max(10, min(60, green_time))
        return int(round(green_time))
    
    def _simple_timing_heuristic(self, vehicle_counts):
        """
        Simple heuristic for signal timing when no model is available
        
        Args:
            vehicle_counts (dict): Counts of different vehicle types
        
        Returns:
            int: Recommended green signal time in seconds
        """
        # Base green time
        base_time = 20
        
        # Adjust time based on vehicle counts
        time_adjustment = (
            vehicle_counts['cars'] * 0.5 +  # Cars have less weight
            vehicle_counts['trucks'] * 1.0 +  # Trucks need more time
            vehicle_counts['buses'] * 1.5 +  # Buses need more time
            vehicle_counts['bikes'] * 0.3    # Bikes need less time
        )
        
        # Calculate final green time
        green_time = base_time + int(time_adjustment)
        
        # Ensure within reasonable bounds
        return max(10, min(60, green_time))

def main():
    # Create output directory
    os.makedirs('output', exist_ok=True)
    
    # Create instances
    phone_camera = PhoneCameraCapture(ip_address='http://192.0.0.4:8080')  # Replace with your IP
    detector = VehicleDetectionSystem()
    optimizer = TrafficSignalOptimizer()
    
    # Try to connect to phone camera
    if phone_camera.connect_camera():
        # Capture image
        input_image_path = 'output/phone_camera_input.jpg'
        if phone_camera.capture_image(input_image_path):
            try:
                # Detect vehicles
                vehicle_counts = detector.detect_vehicles(input_image_path)
                
                # Visualize detections
                detection_output_path = 'output/detected_vehicles.jpg'
                detector.visualize_detections(input_image_path, detection_output_path)
                
                # Predict signal timing
                green_time = optimizer.predict_signal_timing(vehicle_counts)
                
                # Print results
                print("\n--- Traffic Analysis Results ---")
                print("Vehicle Counts:", vehicle_counts)
                print(f"Recommended Green Signal Time: {green_time} seconds")
                print(f"Input Image: {input_image_path}")
                print(f"Detection Visualization: {detection_output_path}")
            
            except Exception as e:
                print(f"Detection or optimization error: {e}")
    else:
        print("Failed to connect to phone camera. Check IP address and app.")

if __name__ == "__main__":
    main()