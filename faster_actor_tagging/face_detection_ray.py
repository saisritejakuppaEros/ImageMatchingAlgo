"""
Face Detection with Ray Parallel Processing

This script uses Ray to parallelize face detection across multiple workers,
significantly speeding up the processing of large image datasets.

OUTPUT STRUCTURE:
-----------------
CSV file with columns:
  - image_path: Path to the image
  - x1, y1, x2, y2: Bounding box coordinates (xyxy format)
  - confidence: Detection confidence score
  - face_id: Unique ID for each detected face (added after processing)

Each row represents one detected face. Images with multiple faces will have multiple rows.
"""

import os
from pathlib import Path
import pandas as pd
from tqdm import tqdm
import ray
from ultralytics import YOLO


def get_image_paths(img_dirpath):
    """Recursively get all image paths from a directory."""
    imgs = []
    for root, dirs, files in os.walk(img_dirpath):
        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg")):
                imgs.append(os.path.join(root, file))
    return imgs


@ray.remote
class FaceDetector:
    """
    Ray Actor that loads the YOLO model once and processes batches of images.
    Using an Actor ensures the model is loaded only once per worker.
    """
    def __init__(self, model_path):
        """Initialize the YOLO model."""
        self.model = YOLO(model_path)
    
    def detect_faces_batch(self, image_batch):
        """
        Detect faces in a batch of images.
        
        Args:
            image_batch: List of image paths
            
        Returns:
            List of dicts with detection results
        """
        data_list = []
        
        # Run YOLO detection on the batch
        results = self.model(image_batch, verbose=False)
        
        for result in results:
            # Create a separate row for each detected face
            if len(result.boxes) > 0:
                for j in range(len(result.boxes)):
                    # Get bounding box in original image coordinates (xyxy format: x1, y1, x2, y2)
                    bbox = result.boxes.xyxy[j].cpu().numpy()
                    conf = result.boxes.conf[j].cpu().item()
                    
                    data_list.append({
                        "image_path": result.path,
                        "x1": float(bbox[0]),
                        "y1": float(bbox[1]),
                        "x2": float(bbox[2]),
                        "y2": float(bbox[3]),
                        "confidence": conf
                    })
        
        return data_list


def detect_faces_ray(imgs, model_path, batch_size, num_workers=None):
    """
    Detect faces using Ray for parallel processing.
    
    Args:
        imgs: List of image paths
        model_path: Path to YOLO model file
        batch_size: Number of images to process per batch
        num_workers: Number of Ray workers (None = use all available CPUs)
        
    Returns:
        List of detection dictionaries
    """
    # Initialize Ray
    if not ray.is_initialized():
        if num_workers is None:
            ray.init()
        else:
            ray.init(num_cpus=num_workers)
    
    print(f"Ray initialized with {ray.available_resources().get('CPU', 0)} CPUs")
    
    # Create Ray actors (one per worker)
    # Use fewer actors than CPUs to avoid oversubscription
    num_actors = min(num_workers or int(ray.available_resources().get('CPU', 1)), len(imgs) // batch_size + 1)
    num_actors = max(1, min(num_actors, 16))  # Cap at 16 actors
    
    print(f"Creating {num_actors} Ray actors with YOLO models...")
    actors = [FaceDetector.remote(model_path) for _ in range(num_actors)]
    
    # Split images into batches and assign to actors
    image_batches = []
    for i in range(0, len(imgs), batch_size):
        batch = imgs[i:i+batch_size]
        image_batches.append(batch)
    
    print(f"Split {len(imgs)} images into {len(image_batches)} batches of size {batch_size}")
    print(f"Submitting tasks to Ray workers...")
    
    # Submit batches to actors in round-robin fashion
    futures = []
    for i, batch in enumerate(image_batches):
        actor = actors[i % num_actors]
        future = actor.detect_faces_batch.remote(batch)
        futures.append(future)
    
    # Collect results with progress bar
    print(f"Processing {len(futures)} batches in parallel...")
    all_detections = []
    for future in tqdm(futures, desc="Processing batches"):
        batch_results = ray.get(future)
        all_detections.extend(batch_results)
    
    return all_detections


def save_results(detections, output_csv):
    """Save detection results to CSV."""
    if detections:
        df = pd.DataFrame(detections)
        df.to_csv(output_csv, index=False)
        print(f"Results saved to: {output_csv}")
        return len(detections)
    else:
        print("No faces detected!")
        return 0


def main(input_dir, output_csv, batch_size, model_path, num_workers=None):
    """
    Main function to run face detection with Ray.
    
    Args:
        input_dir: Directory containing images
        output_csv: Output CSV file path
        batch_size: Number of images to process per batch
        model_path: Path to YOLO model file
        num_workers: Number of Ray workers (None = use all available CPUs)
    """
    print(f"{'='*60}")
    print(f"RAY-ACCELERATED FACE DETECTION")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Input directory: {input_dir}")
    print(f"  Output CSV: {output_csv}")
    print(f"  Model: {model_path}")
    print(f"  Batch size: {batch_size}")
    print(f"  Num workers: {num_workers if num_workers else 'Auto (all CPUs)'}")
    print(f"{'='*60}\n")
    
    # Get all images
    print(f"Scanning directory: {input_dir}")
    imgs = get_image_paths(input_dir)
    print(f"Total images found: {len(imgs)}")
    
    if len(imgs) == 0:
        print("No images found! Exiting.")
        return
    
    # Detect faces with Ray
    print(f"\nStarting face detection with Ray...")
    detections = detect_faces_ray(imgs, model_path, batch_size, num_workers)
    
    # Save results
    print(f"\nSaving results...")
    total_faces = save_results(detections, output_csv)
    
    print(f"\n{'='*60}")
    print(f"Detection complete!")
    print(f"Total faces detected: {total_faces}")
    print(f"Total images processed: {len(imgs)}")
    print(f"Average faces per image: {total_faces / len(imgs):.2f}")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*60}")
    
    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    # Configuration variables - edit these as needed
    input_dir = '/data0/PArun/Quality_checklist/dataset/cropped_detected_shots/Sep/Devdas'
    output_csv = '/data0/teja_works/dataset_captioning/faster_actor_tagging/devadas_faces.csv'
    batch_size = 32  # Number of images per batch
    model_path = 'yolov12m-face.pt'
    cpu_usage_percentage = 50  # Percentage of available CPUs to use (1-100)
    
    # Calculate number of workers based on CPU percentage
    import multiprocessing
    total_cpus = multiprocessing.cpu_count()
    num_workers = max(1, int(total_cpus * (cpu_usage_percentage / 100.0)))
    print(f"Using {num_workers} out of {total_cpus} available CPUs ({cpu_usage_percentage}%)")
    
    main(input_dir, output_csv, batch_size, model_path, num_workers)

