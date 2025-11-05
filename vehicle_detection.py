import cv2
import numpy as np
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from PIL import Image
import matplotlib.pyplot as plt
import pandas as pd
import time

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
            'rickshaws': 0 ,
             'road width':15 # Note: No direct rickshaw detection
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
                box = box.cpu().numpy()
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
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
    
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

def main():
    # Create detection and optimization instances
    detector = VehicleDetectionSystem()
    optimizer = TrafficSignalOptimizer()
    
    # Example usage
    image_path = 'input/input7.jpg'  # Replace with your image path
    
    try:
        # Detect vehicles
        vehicle_counts = detector.detect_vehicles(image_path)
        
        # Visualize detections
        detector.visualize_detections(image_path)
        
        # Predict signal timing
        green_time = optimizer.predict_signal_timing(vehicle_counts)
        
        print("Vehicle Counts:", vehicle_counts)
        print(f"Recommended Green Signal Time: {green_time} seconds")
    
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    main()