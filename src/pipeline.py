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
        image,
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
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Unsupported input type: {type(image)}")
        
        cv_image = image.copy()

        # Detect monuments
        detections = self.detector.detect(cv_image)
        
        # Process each detection
        results = []
        for det in detections:
            bbox = det['bbox']
            confidence = det['conf']
            
            # Extract region of interest
            roi = extract_roi(cv_image, bbox)
            
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
                    2
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
                        2, 
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
        
    def _merge_duplicate_detections(self, detections, iou_threshold=0.5, landmark_threshold=0.5):
        """
        Merge multiple detections of the same landmark.
        
        Args:
            detections: List of detection results
            iou_threshold: IoU threshold to consider bounding boxes as overlapping
            landmark_threshold: Score threshold to consider landmarks as the same
            
        Returns:
            List of merged detections
        """
        if not detections:
            return []
            
        # Sort detections by confidence (highest first)
        sorted_dets = sorted(detections, key=lambda x: x['confidence'], reverse=True)
        
        # Initialize list of merged detections
        merged_dets = []
        used_indices = set()
        
        for i, det in enumerate(sorted_dets):
            if i in used_indices:
                continue
                
            current_det = det.copy()
            current_bbox = det['bbox']
            current_landmark = det['matches'][0]['landmark'] if det['matches'] else None
            
            # Skip if no landmark was matched
            if not current_landmark:
                merged_dets.append(current_det)
                used_indices.add(i)
                continue
            
            # Find all detections of the same landmark
            same_landmarks = []
            for j, other_det in enumerate(sorted_dets):
                if j in used_indices or j == i:
                    continue
                    
                other_bbox = other_det['bbox']
                other_landmark = other_det['matches'][0]['landmark'] if other_det['matches'] else None
                
                # Skip if no landmark match
                if not other_landmark:
                    continue
                
                # Check if it's the same landmark
                if current_landmark == other_landmark:
                    # Calculate IoU
                    iou = self._calculate_iou(current_bbox, other_bbox)
                    # Similar landmark and overlapping bounding boxes
                    if iou >= iou_threshold:
                        same_landmarks.append(j)
            
            # If we found duplicates, merge them
            if same_landmarks:
                # Mark all as used
                used_indices.add(i)
                used_indices.update(same_landmarks)
                
                # Keep the highest confidence detection's bbox
                # No need to modify current_det's bbox
                
                # Combine matches by selecting unique landmarks with highest scores
                all_matches = current_det['matches'].copy()
                for idx in same_landmarks:
                    all_matches.extend(sorted_dets[idx]['matches'])
                
                # Group by landmark and keep the highest score for each
                by_landmark = {}
                for match in all_matches:
                    landmark = match['landmark']
                    if landmark not in by_landmark or match['score'] > by_landmark[landmark]['score']:
                        by_landmark[landmark] = match
                
                # Sort by score (descending)
                merged_matches = sorted(by_landmark.values(), key=lambda x: x['score'], reverse=True)
                current_det['matches'] = merged_matches
                
                merged_dets.append(current_det)
            else:
                # No duplicates found, add the current detection
                merged_dets.append(current_det)
                used_indices.add(i)
        
        # Add any remaining detections that weren't merged
        for i, det in enumerate(sorted_dets):
            if i not in used_indices:
                merged_dets.append(det)
                used_indices.add(i)
        
        return merged_dets
    
    def _calculate_iou(self, bbox1, bbox2):
        """
        Calculate Intersection over Union (IoU) between two bounding boxes.
        
        Args:
            bbox1: First bounding box [x1, y1, x2, y2]
            bbox2: Second bounding box [x1, y1, x2, y2]
            
        Returns:
            IoU value (0.0 to 1.0)
        """
        # Extract coordinates
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        # Calculate intersection area
        x_left = max(x1_1, x1_2)
        y_top = max(y1_1, y1_2)
        x_right = min(x2_1, x2_2)
        y_bottom = min(y2_1, y2_2)
        
        if x_right < x_left or y_bottom < y_top:
            return 0.0
            
        intersection_area = (x_right - x_left) * (y_bottom - y_top)
        
        # Calculate union area
        bbox1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        bbox2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        union_area = bbox1_area + bbox2_area - intersection_area
        
        # Calculate IoU
        if union_area == 0:
            return 0.0
            
        return intersection_area / union_area
    
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

    def visualize_results(self, image, results, output_path=None, show_all_matches=False):
        """
        Create a visualization of detection and recognition results.
        
        Args:
            image_path: Path to the original image
            results: Results from process_image
            output_path: Path to save the visualization (optional)
            show_all_matches: Whether to show all matches or just the top one
            
        Returns:
            Annotated image (numpy array)
        """
        if not isinstance(image, np.ndarray):
            raise TypeError(f"Unsupported input type: {type(image)}")
        
        # Colors for different detections
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue (BGR format)
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (0, 255, 255),  # Yellow
            (255, 0, 255)   # Magenta
        ]
        
        # Draw each detection
        for i, det in enumerate(results['detections']):
            color = colors[i % len(colors)]
            bbox = det['bbox']
            confidence = det['confidence']
            matches = det['matches']
            
            # Draw bounding box
            cv2.rectangle(
                image, 
                (bbox[0], bbox[1]), 
                (bbox[2], bbox[3]), 
                color, 
                2
            )
            
            # Add detection info
            det_info = f"Det {i+1}: {confidence:.2f}"
            cv2.putText(
                image, 
                det_info, 
                (bbox[0], bbox[1] - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                2, 
                color, 
                2
            )
            
            # Add match information
            if matches and len(matches) > 0:
                if show_all_matches:
                    # Show all matches with scores
                    for j, match in enumerate(matches):
                        match_text = f"{match['landmark']} ({match['score']:.2f})"
                        y_pos = bbox[3] + 20 + (j * 20)  # Position below the bbox
                        cv2.putText(
                            image, 
                            match_text, 
                            (bbox[0], y_pos), 
                            cv2.FONT_HERSHEY_SIMPLEX, 
                            2, 
                            color, 
                            2
                        )
                else:
                    # Show only the top match
                    top_match = matches[0]
                    match_text = f"{top_match['landmark']} ({top_match['score']:.2f})"
                    cv2.putText(
                        image, 
                        match_text, 
                        (bbox[0], bbox[3] + 20), 
                        cv2.FONT_HERSHEY_SIMPLEX, 
                        2, 
                        color, 
                        2
                    )
        
        # Save image if output path is provided
        if output_path:
            cv2.imwrite(output_path, image)
            print(f"Visualization saved to {output_path}")
        
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