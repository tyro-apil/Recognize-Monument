"""
Local feature extraction implementation.
"""

import torch
from lightglue import SuperPoint
from lightglue.utils import load_image

class LocalFeatureExtractor:
    """
    Local feature extractor using SuperPoint.
    """
    def __init__(self, max_kp: int = 1024, device: str = None):
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.model = SuperPoint(max_num_keypoints=max_kp).eval().to(device)
    
    def extract(self, img_path):
        """
        Extract local features (keypoints and descriptors) from an image.
        
        Args:
            image: Input image (PIL Image, numpy array, or tensor)
            
        Returns:
            features
        """
        image = load_image(img_path)
        feats = self.model.extract(image)
        
        return feats
    