import os
import cv2
from src.pipeline import MonumentPipeline

def main():
    """Example usage of the MonumentPipeline."""

    image_path="./images/img1.jpg"
    model="./weights/yolov11_det.engine"
    confidence=0.7
    milvus_uri="./data/monumentdb.db"
    device="cuda"

    top_k=5
    show_all=False
    output="./results"
    
    # Create output directory if it doesn't exist
    os.makedirs(output, exist_ok=True)
    
    image = cv2.imread(image_path)
    # Get base filename without extension
    base_name = os.path.splitext(os.path.basename(image))[0]
    
    # Initialize pipeline
    pipeline = MonumentPipeline(
        detector_model_path=model,
        detector_confidence=confidence,
        milvus_uri=milvus_uri,
        device=device
    )
    
    # Process the image
    results = pipeline.process_image(
        image_path=image,
        top_k=top_k,
        return_image=True
    )
    
    # Create and save visualization
    output_image_path = os.path.join(output, f"{base_name}_result.jpg")
    pipeline.visualize_results(
        image_path=image,
        results=results,
        output_path=output_image_path,
        show_all_matches=show_all
    )
    
    # Save JSON results
    json_results = pipeline.to_json(results)
    output_json_path = os.path.join(output, f"{base_name}_result.json")
    with open(output_json_path, 'w') as f:
        f.write(json_results)
    
    print(f"Results saved to {output}/")


if __name__ == "__main__":
    main()