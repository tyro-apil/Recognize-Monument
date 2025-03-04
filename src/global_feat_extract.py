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

class GlobalFeatureExtractor():
    def __init__(self, model_name='efficientnet_b3', pretrained=True, gem_p=3, device=None):
        super(GlobalFeatureExtractor, self).__init__()
        
        self.device = device if device is not None else torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Create the base model without classifier and with empty pooling
        self.backbone = timm.create_model(
            model_name,
            pretrained=pretrained,
            num_classes=0,
            global_pool=''              # Remove default pooling
        ).to(self.device)
        
        self.gem_pooling = GeM(p=gem_p).to(self.device)
        
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),  # Fixed size for all images
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
    def forward(self, x):
        features = self.backbone(x)
        pooled_features = self.gem_pooling(features)
        return pooled_features.view(x.size(0), -1)
    
    def extract(self, image):
        """
        Extract global features from an image.
        Args:
            image: Input image (PIL Image, numpy array, or tensor)
        Returns:
            Feature tensor of shape [batch_size, feature_dim]
        """
        # Handle different input types
        if not isinstance(image, torch.Tensor):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            image = self.transform(image).unsqueeze(0)
        elif image.dim() == 3:  
            image = image.unsqueeze(0)
        
        image = image.to(self.device)
        
        features = self.forward(image)
        
        return features