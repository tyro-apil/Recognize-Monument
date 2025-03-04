import numpy as np
from PIL import Image

def extract_roi(image, bbox):
    """
    Extract a region of interest (ROI) from an image using bounding box coordinates.
    
    Args:
        image: Input image (PIL.Image or numpy array from cv2)
        bbox: Bounding box coordinates in xyxy format [x1, y1, x2, y2]
    
    Returns:
        ROI image in the same format as input
    """
    # Get bbox coordinates
    x1, y1, x2, y2 = map(int, bbox)  # Convert to integers
    
    # Handle PIL Image
    if isinstance(image, Image.Image):
        width, height = image.size
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        return image.crop((x1, y1, x2, y2))
    
    # Handle numpy array (cv2 image)
    elif isinstance(image, np.ndarray):
        height, width = image.shape[:2]
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(width, x2)
        y2 = min(height, y2)
        return image[y1:y2, x1:x2]
    
    # Handle unsupported types
    else:
        raise TypeError(f"Unsupported image type: {type(image)}")