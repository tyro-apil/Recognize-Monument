"""
src/detect.py

YOLOv11 monument detector implementation.
"""

from ultralytics import YOLO
from typing import List, Dict, Any
import cv2

class YOLOMonumentDetector:
    """
    Monument detector using YOLOv11.
    
    This model is responsible for identifying potential monument regions
    in an image with bounding boxes and confidence scores.
    """
    
    def __init__(self, model_path: str, confidence_threshold: float = 0.4, device: str = 'cuda'):
        self.device = device
        self.confidence_threshold = confidence_threshold
        
        self.model = YOLO(model=f"{model_path}")
    
    def detect(self, image) -> List[Dict[str, Any]]:
        """
        Detect monuments in an image.
        
        Args:
            image: Input image (PIL Image, numpy array, or file path)
            
        Returns:
            List of detection results, each with bbox, confidence
        """
        results = self.model.predict(image)
        result = results[0]
        
        detections = []

        for i, box in enumerate(result.boxes):
            # print(box.conf)
            # print(box.xyxy[0])

            if box.conf >= self.confidence_threshold:
                detections.append({
                    'id': i,
                    'bbox': [int(box.xyxy[0][0]), int(box.xyxy[0][1]), int(box.xyxy[0][2]), int(box.xyxy[0][3])],
                    'conf': box.conf.cpu().numpy()[0],
                })

        return detections