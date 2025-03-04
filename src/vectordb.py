"""
image_indexer.py
Add image feature vectors to Milvus database
"""
import os
import torch
import re
from tqdm import tqdm
from pymilvus import MilvusClient
from .global_feat_extract import GlobalFeatureExtractor

class MilvusImageIndexer:
    def __init__(self, 
                 milvus_uri="./data/vectors.db", 
                 collection_name="image_features",
                 model_name="efficientnet_b3",
                 gem_p=3,
                 device=None):
        """Initialize the Milvus Image Indexer"""
        # Ensure directory exists
        os.makedirs(os.path.dirname(os.path.abspath(milvus_uri)), exist_ok=True)
        
        # Initialize Milvus client
        self.client = MilvusClient(milvus_uri)
        self.collection_name = collection_name
        
        # Initialize feature extractor
        self.extractor = GlobalFeatureExtractor(
            model_name=model_name,
            pretrained=True,
            gem_p=gem_p,
            device=device
        )
        
        # Get the feature dimension from the model
        dummy_input = torch.zeros(1, 3, 224, 224).to(self.extractor.device)
        with torch.no_grad():
            dummy_output = self.extractor(dummy_input)
        self.feature_dim = dummy_output.shape[1]
        
        # Create collection if it doesn't exist
        if not self.client.has_collection(self.collection_name):
            print(f"Creating collection '{self.collection_name}' with dimension {self.feature_dim}")
            self.client.create_collection(
                collection_name=self.collection_name,
                dimension=self.feature_dim
            )
    
    def _get_next_id(self):
        """Get the next available ID for inserting documents"""
        try:
            result = self.client.query(
                collection_name=self.collection_name,
                output_fields=["id"],
                limit=1,
                sort="id desc"
            )
            return int(result[0]["id"]) + 1 if result else 1
        except Exception:
            return 1  # Default to 1 if there's an error
    
    def _extract_landmark_name(self, filename):
        """Extract landmark name from the filename, e.g., krishna_mandir1_1.jpg -> krishna_mandir"""
        match = re.match(r'([a-zA-Z_]+\d*)', filename)
        if match:
            base_name = match.group(1)
            landmark_name = re.sub(r'\d+$', '', base_name).rstrip('_')
            return landmark_name
        return None
    
    def index_directory(self, image_dir, batch_size=32):
        """Index all images in a directory"""
        # Get list of image files
        image_files = [f for f in os.listdir(image_dir) 
                       if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        
        if not image_files:
            print(f"No images found in {image_dir}")
            return 0
        
        print(f"Found {len(image_files)} images in {image_dir}")
        
        # Get the next available ID
        next_id = self._get_next_id()
        print(f"Starting with ID: {next_id}")
        
        # Process images in batches
        indexed_count = 0
        for i in range(0, len(image_files), batch_size):
            batch_files = image_files[i:i+batch_size]
            batch_data = []
            
            # Process batch
            for idx, filename in enumerate(tqdm(batch_files, desc=f"Batch {i//batch_size + 1}")):
                image_path = os.path.join(image_dir, filename)
                
                try:
                    # Extract features
                    features = self.extractor.extract(image_path)
                    feature_list = features.squeeze(0).cpu().tolist()
                    
                    # Extract landmark name
                    landmark_name = self._extract_landmark_name(filename)
                    
                    # Prepare data for Milvus
                    batch_data.append({
                        "id": next_id + idx,
                        "vector": feature_list,
                        "filename": filename,
                        "landmark": landmark_name
                    })
                    
                except Exception as e:
                    print(f"Error processing {filename}: {e}")
            
            # Insert batch into Milvus
            if batch_data:
                try:
                    result = self.client.insert(
                        collection_name=self.collection_name,
                        data=batch_data
                    )
                    indexed_count += result['insert_count']
                    next_id += len(batch_data)  # Update next_id for the next batch
                    print(f"Inserted {result['insert_count']} vectors")
                except Exception as e:
                    print(f"Error inserting batch: {e}")
        
        return indexed_count
    
    def search_similar(self, image_path, top_k=3):
        """Search for similar images in the database"""
        features = self.extractor.extract(image_path)
        feature_list = features.squeeze(0).cpu().tolist()
        
        return self.client.search(
            collection_name=self.collection_name,
            data=[feature_list],
            limit=top_k,
            output_fields=["filename", "landmark"]
        )
    
    def search_by_landmark(self, landmark_name):
        """Search for images by landmark name"""
        return self.client.query(
            collection_name=self.collection_name,
            filter=f"landmark == '{landmark_name}'",
            output_fields=["id", "filename", "landmark"]
        )
    
    def get_all_landmarks(self):
        """Get a list of all unique landmarks in the database"""
        results = self.client.query(
            collection_name=self.collection_name,
            output_fields=["landmark"]
        )
        
        landmarks = set()
        for item in results:
            if 'landmark' in item and item['landmark']:
                landmarks.add(item['landmark'])
        
        return sorted(list(landmarks))