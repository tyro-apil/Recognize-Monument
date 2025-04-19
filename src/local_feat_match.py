"""
src/local_feat_match
Local feature matching implementation using LightGlue.
"""
import torch
import numpy as np
from pathlib import Path

# Import LightGlue components
from libs.LightGlue.lightglue import SuperPoint, LightGlue
from libs.LightGlue.lightglue.utils import load_image
from libs.LightGlue.lightglue.viz2d import plot_matches

class LocalMatcher:
    """
    Local feature matching with LightGlue and SuperPoint.
    """
    
    def __init__(self, device=None):
        """
        Initialize local feature matcher.
        
        Args:
            device: Device to run models on ('cuda', 'cpu', etc.)
        """
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device
            
        print(f"Local matcher using device: {self.device}")
        
        self.extractor = SuperPoint(max_num_keypoints=2048).eval().to(self.device)
        self.matcher = LightGlue(features='superpoint').eval().to(self.device)
    
    def extract_features(self, image):
        """
        Extract local features from an image.
        
        Args:
            image: Image as a numpy array or path to image file
            
        Returns:
            extracted features and loaded image tensor
        """
        # Load image based on input type
        if isinstance(image, str) or isinstance(image, Path):
            # Load from path
            img_tensor = load_image(image).to(self.device)
        elif isinstance(image, np.ndarray):
            # Convert numpy array to RGB if needed
            if image.ndim == 3 and image.shape[2] == 3:
                # Check if BGR (OpenCV) and convert to RGB
                if np.mean(image[:, :, 0]) < np.mean(image[:, :, 2]):
                    image = image[:, :, ::-1]  # BGR to RGB
            
            # Convert to torch tensor
            img_tensor = torch.from_numpy(image).float().permute(2, 0, 1).unsqueeze(0)
            img_tensor = img_tensor / 255.0  # Normalize to [0,1]
            img_tensor = img_tensor.to(self.device)
        else:
            raise TypeError(f"Unsupported image type: {type(image)}")
        
        # Extract features
        with torch.no_grad():
            feats = self.extractor(img_tensor)
        
        return feats, img_tensor
    
    def match_and_score(self, feats0, feats1):
        """
        Match features between two images and calculate matching score.
        
        Args:
            feats0: Features from the first image
            feats1: Features from the second image
            
        Returns:
            Dictionary with matching statistics
        """
        # Match features
        with torch.no_grad():
            matches01 = self.matcher({'image0': feats0, 'image1': feats1})
            matches01 = {k: v[0] for k, v in matches01.items()}
        
        # Calculate metrics
        num_matches = len(matches01['matches'])
        
        # Match confidence (average score of all matches)
        match_confidence = matches01['scores'].mean().item() if num_matches > 0 else 0
        
        # Calculate inlier ratio
        # Higher values indicate better quality matches
        inlier_ratio = num_matches / max(len(feats0['keypoints'][0]), 1)
        
        # Calculate matching score (custom metric)
        # Combines number of matches and confidence
        matching_score = num_matches * match_confidence
        
        return {
            'num_matches': num_matches,
            'match_confidence': match_confidence,
            'inlier_ratio': inlier_ratio,
            'matching_score': matching_score,
            'raw_matches': matches01
        }
    
    def visualize_matches(self, image0, image1, feats0, feats1, matches):
        """
        Visualize matches between two images.
        
        Args:
            image0: First image tensor
            image1: Second image tensor
            feats0: Features from first image
            feats1: Features from second image
            matches: Matching results
            
        Returns:
            Matplotlib axes object with visualization
        """
        # Convert to numpy
        img0_np = image0.cpu().squeeze(0).permute(1, 2, 0).numpy()
        img1_np = image1.cpu().squeeze(0).permute(1, 2, 0).numpy()
        
        # Get keypoints and matches
        kpts0 = feats0['keypoints'][0].cpu().numpy()
        kpts1 = feats1['keypoints'][0].cpu().numpy()
        matches_np = matches['raw_matches']['matches'].cpu().numpy()
        scores = matches['raw_matches']['scores'].cpu().numpy()
        
        # Generate visualization
        axes = plot_matches(
            img0_np, 
            img1_np,
            kpts0,
            kpts1,
            matches_np,
            scores,
            title=f'Matches: {matches["num_matches"]}, Score: {matches["matching_score"]:.2f}'
        )
        
        return axes

def rerank_with_local_features(query_image, candidate_images, local_matcher, top_k=None):
    """
    Rerank candidate images based on local feature matching.
    
    Args:
        query_image: Query image (numpy array or path)
        candidate_images: List of (path, metadata) tuples for candidate images
        local_matcher: LocalMatcher instance
        top_k: Number of top candidates to return (None = all)
        
    Returns:
        List of reranked candidates with scores
    """
    # Extract features from query image
    query_feats, query_img = local_matcher.extract_features(query_image)
    
    # Match with each candidate
    reranked_results = []
    
    for candidate_info in candidate_images:
        # Extract candidate path and metadata
        candidate_path = candidate_info['filename']
        
        try:
            # Extract features from candidate
            candidate_feats, candidate_img = local_matcher.extract_features(candidate_path)
            
            # Match features and calculate score
            match_info = local_matcher.match_and_score(query_feats, candidate_feats)
            
            # Add to results with original metadata plus matching info
            result = {
                **candidate_info,  # Original metadata (landmark, filename, global_score)
                'local_matches': match_info['num_matches'],
                'match_confidence': match_info['match_confidence'],
                'inlier_ratio': match_info['inlier_ratio'],
                'local_score': match_info['matching_score']
            }
            
            reranked_results.append(result)
            
        except Exception as e:
            print(f"Error processing {candidate_path}: {e}")
            # Add with zero score
            result = {
                **candidate_info,
                'local_matches': 0,
                'match_confidence': 0,
                'inlier_ratio': 0,
                'local_score': 0
            }
            reranked_results.append(result)
    
    # Sort by local matching score (descending)
    reranked_results = sorted(reranked_results, key=lambda x: x['local_score'], reverse=True)
    
    # Return top_k if specified
    if top_k is not None:
        return reranked_results[:top_k]
    
    return reranked_results