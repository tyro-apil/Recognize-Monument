import cv2
from src.pipeline import MonumentPipeline

def main():
    """Example usage of the MonumentPipeline."""
    # Initialize pipeline
    pipeline = MonumentPipeline(
        detector_model_path="./weights/yolov11_det.engine",
        milvus_uri="./data/monumentdb.db"
    )
    
    # Process an image
    image_path = "./images/img3.jpg"
    results = pipeline.process_image(
        image_path=image_path,
        top_k=3,
        return_image=True
    )
    
    # Print results
    print(f"Found {results['num_detections']} monuments")
    
    # Convert to JSON
    json_results = pipeline.to_json(results)
    print(json_results)
    
    # Save visualization if available
    if 'image' in results:
        cv2.imwrite("./data/result.jpg", results['image'])
        print("Visualization saved to ./data/result.jpg")


if __name__ == "__main__":
    main()