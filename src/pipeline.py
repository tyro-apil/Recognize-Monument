"""
src/pipeline.py

Monument detection and recognition pipeline.
Integrates detection, feature extraction, and database matching.
"""

import json
from typing import List, Dict, Any, Union, Optional
import torch
import numpy as np
from PIL import Image
import cv2

from .detect import YOLOMonumentDetector
from .vectordb import MilvusImageIndexer
from utils.preprocess import extract_roi

class MonumentPipeline:
    """
    End-to-end pipeline for monument detection and recognition.
    
    This pipeline integrates:
    1. Monument detection using YOLOv11
    2. Feature extraction from detected regions
    3. Matching against a database of known monuments
    """
    
    def __init__(
        self,
        detector_model_path: str,
        detector_confidence: float = 0.7,
        extractor_model_name: str = "efficientnet_b3",
        milvus_uri: str = "./data/monumentdb.db",
        collection_name: str = "global_features",
        device: Optional[str] = None
    ):
        """
        Initialize the monument detection and recognition pipeline.
        
        Args:
            detector_model_path: Path to the YOLOv11 detector model
            detector_confidence: Confidence threshold for detections
            extractor_model_name: Name of the feature extractor model
            milvus_uri: URI for the Milvus database
            collection_name: Name of the collection in Milvus
            device: Device to run models on ('cuda', 'cpu', etc.)
        """
        # Determine device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Using device: {self.device}")
        
        # Initialize detector
        self.detector = YOLOMonumentDetector(
            model_path=detector_model_path,
            confidence_threshold=detector_confidence,
            device=self.device
        )
        
        # Initialize database connection and feature extractor
        self.db = MilvusImageIndexer(
            milvus_uri=milvus_uri,
            collection_name=collection_name,
            model_name=extractor_model_name,
            device=self.device
        )
        
        # Get reference to the feature extractor from the DB indexer
        self.feature_extractor = self.db.extractor
    
    def process_image(
        self, 
        image_path: str,
        top_k: int = 1,
        return_image: bool = False
    ) -> Dict[str, Any]:
        """
        Process an image to detect and recognize monuments.
        
        Args:
            image_path: Path to the input image
            top_k: Number of top matches to return for each detection
            return_image: Whether to return the annotated image
            
        Returns:
            Dictionary with detection and recognition results
        """
        # Load image
        if isinstance(image_path, str):
            # Load with PIL for feature extraction
            pil_image = Image.open(image_path).convert('RGB')
            # Load with OpenCV for detection and visualization
            cv_image = cv2.imread(image_path)
        else:
            raise TypeError(f"Unsupported input type: {type(image_path)}")
        
        # Detect monuments
        detections = self.detector.detect(cv_image)
        
        # Process each detection
        results = []
        for det in detections:
            bbox = det['bbox']
            confidence = det['conf']
            
            # Extract region of interest
            roi = extract_roi(pil_image, bbox)
            
            # Search for matches in the database
            matches = self._search_similar_monuments(roi, top_k)
            
            # Format the result
            result = {
                'bbox': bbox,
                'confidence': float(confidence),
                'matches': matches
            }
            results.append(result)
            
            # Draw on image if requested
            if return_image:
                # Draw bounding box
                cv2.rectangle(
                    cv_image, 
                    (bbox[0], bbox[1]), 
                    (bbox[2], bbox[3]), 
                    (0, 255, 0), 
                    4
                )
                
                # Add text for top match if available
                if matches and len(matches) > 0:
                    top_match = matches[0]
                    label = f"{top_match['landmark']} ({top_match['score']:.2f})"
                    cv2.putText(
                        cv_image, 
                        label, 
                        (bbox[0], bbox[1] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        1.5, 
                        (0, 255, 0), 
                        2
                    )
        
        # Prepare response
        response = {
            'num_detections': len(results),
            'detections': results
        }
        
        if return_image:
            response['image'] = cv_image
        
        return response
    
    def _search_similar_monuments(self, roi_image, top_k=1):
        """
        Search for similar monuments in the database.
        
        Args:
            roi_image: Region of interest image (PIL Image)
            top_k: Number of top matches to return
            
        Returns:
            List of matching monuments with scores
        """
        # Extract features from ROI
        features = self.feature_extractor.extract(roi_image)
        feature_list = features.squeeze(0).cpu().tolist()
        
        # Search in database
        search_results = self.db.client.search(
            collection_name=self.db.collection_name,
            data=[feature_list],
            limit=top_k,
            output_fields=["filename", "landmark"]
        )
        
        # Format results
        matches = []
        if search_results and len(search_results) > 0:
            for hit in search_results[0]:
                matches.append({
                    'landmark': hit['entity']['landmark'],
                    'filename': hit['entity']['filename'],
                    'score': hit['distance']
                })
        
        return matches

    def visualize_results(self, image_path, results):
        """
        Create a visualization of detection and recognition results.
        
        Args:
            image_path: Path to the original image
            results: Results from process_image
            
        Returns:
            Annotated image (numpy array)
        """
        # Load image
        image = cv2.imread(image_path)
        
        # Draw each detection
        for det in results['detections']:
            bbox = det['bbox']
            matches = det['matches']
            
            # Draw bounding box
            cv2.rectangle(
                image, 
                (bbox[0], bbox[1]), 
                (bbox[2], bbox[3]), 
                (0, 255, 0), 
                2
            )
            
            # Add text for top match if available
            if matches and len(matches) > 0:
                top_match = matches[0]
                label = f"{top_match['landmark']} ({top_match['score']:.2f})"
                cv2.putText(
                    image, 
                    label, 
                    (bbox[0], bbox[1] - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 
                    0.5, 
                    (0, 255, 0), 
                    2
                )
        
        return image
    
    def to_json(self, results):
        """
        Convert results to JSON string.
        
        Args:
            results: Results from process_image
            
        Returns:
            JSON string
        """
        # Create a copy without the image field
        json_results = {k: v for k, v in results.items() if k != 'image'}
        
        return json.dumps(json_results, indent=2)


