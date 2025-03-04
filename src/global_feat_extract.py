"""
global_feat_extract.py
Global feature extractor
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import timm
from torchvision import transforms
from PIL import Image
import numpy as np
from timm.data import resolve_data_config
from timm.data.transforms_factory import create_transform

class GeM(nn.Module):
    """
    Generalized Mean Pooling (GeM) implementation.
    GeM pooling has been shown to be superior to average pooling for landmarks
    and place recognition tasks.
    """
    def __init__(self, p=3, eps=1e-6):
        super(GeM, self).__init__()
        self.p = nn.Parameter(torch.ones(1) * p)
        self.eps = eps
        
    def forward(self, x):
        return F.avg_pool2d(x.clamp(min=self.eps).pow(self.p),
                           (x.size(-2), x.size(-1))).pow(1./self.p)

class GlobalFeatureExtractor(nn.Module):
    def __init__(self, model_name='efficientnet_b3', pretrained=True, gem_p=3, device=None):
        super(GlobalFeatureExtractor, self).__init__()
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create the base model without classifier and with empty pooling
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''  # Remove default pooling
        ).to(self.device)
        
        self.gem_pooling = GeM(p=gem_p).to(self.device)
        
        # Get model-specific configuration
        config = resolve_data_config({}, model=model_name)
        self.transform = create_transform(**config)
        
        # Set model to evaluation mode
        self.eval()
        
    def forward(self, x):
        with torch.no_grad():
            features = self.backbone(x)
            pooled_features = self.gem_pooling(features)
            # Flatten to get a feature vector
            return pooled_features.view(x.size(0), -1)
    
    def extract(self, image_or_path):
        """
        Extract global features from an image.
        
        Args:
            image_or_path: Input image (PIL Image, numpy array, path string, or tensor)
            
        Returns:
            Normalized feature vector
        """
        # Handle different input types
        if isinstance(image_or_path, str):
            # It's a file path
            image = Image.open(image_or_path).convert('RGB')
            image = self.transform(image).unsqueeze(0)
        elif isinstance(image_or_path, np.ndarray):
            # It's a numpy array
            image = Image.fromarray(image_or_path)
            image = self.transform(image).unsqueeze(0)
        elif isinstance(image_or_path, Image.Image):
            # It's a PIL Image
            image = self.transform(image_or_path).unsqueeze(0)
        elif isinstance(image_or_path, torch.Tensor):
            # It's already a tensor
            image = image_or_path
            if image.dim() == 3:
                image = image.unsqueeze(0)
        else:
            raise TypeError(f"Unsupported input type: {type(image_or_path)}")
            
        # Move to device and extract features
        image = image.to(self.device)
        features = self.forward(image)
        
        # Normalize the feature vector
        features = F.normalize(features, p=2, dim=1)
        
        return features