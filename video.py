from src.pipeline import MonumentPipeline
import cv2

def main():
    """Example usage of the MonumentPipeline."""
    cap=cv2.VideoCapture("./images/patan1.mp4")

    model="./weights/yolov11_det.engine"
    confidence=0.7
    milvus_uri="./data/monumentdb.db"
    device="cuda"

    top_k=5
    show_all=False
    
    
    # Initialize pipeline
    pipeline = MonumentPipeline(
        detector_model_path=model,
        detector_confidence=confidence,
        milvus_uri=milvus_uri,
        device=device
    )

    ret=True

    while ret:
        ret, frame = cap.read()

        if not ret:
            break
    
        # Process the image
        results = pipeline.process_image(
            image=frame,
            top_k=top_k,
            return_image=True
        )
        
        # Create and save visualization
        annotated_img = pipeline.visualize_results(
            image=frame,
            results=results,
            show_all_matches=show_all
        )

        cv2.imshow("frame", annotated_img)

        if cv2.waitKey(33) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()
    


if __name__ == "__main__":
    main()