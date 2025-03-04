"""
index_imgs.py

Index images to db or search in db
"""

from src.vectordb import MilvusImageIndexer
import os

# def main():
# Create data directory if it doesn't exist
os.makedirs("./data", exist_ok=True)

image_dir = "./data/images"
milvus_uri = "./data/monumentdb.db"  # Add .db extension
collection = "global_features"
model = "efficientnet_b3"
batch_size = 32

# Initialize indexer
indexer = MilvusImageIndexer(
    milvus_uri=milvus_uri,
    collection_name=collection,
    model_name=model
)

# Index images
# count = indexer.index_directory(image_dir, batch_size=batch_size)
# print(f"Successfully indexed {count} images")

# similars = indexer.search_similar(image_path="./images/img1.jpg")
similars = indexer.search_by_landmark("krishna_mandir")
print(similars)
