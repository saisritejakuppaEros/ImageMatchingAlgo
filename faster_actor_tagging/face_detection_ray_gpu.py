"""
Face Detection with Ray GPU Parallel Processing

This script uses Ray to parallelize face detection across multiple GPUs,
significantly speeding up the processing of large image datasets.

GPU CONFIGURATION:
------------------
- Specify which GPUs to use via gpu_ids (e.g., [0, 1, 2, 3] or [1, 4, 5, 6])
- Each GPU gets its own Ray actor with a dedicated YOLO model
- Batches are distributed evenly across all specified GPUs

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
import torch


def get_image_paths(img_dirpath):
    """Recursively get all image paths from a directory."""
    imgs = []
    for root, dirs, files in os.walk(img_dirpath):
        for file in files:
            if file.endswith((".jpg", ".png", ".jpeg")):
                imgs.append(os.path.join(root, file))
    return imgs


@ray.remote(num_gpus=1)
class GPUFaceDetector:
    """
    Ray Actor that loads the YOLO model once on a specific GPU and processes batches.
    Each actor is assigned to one GPU using Ray's GPU resource management.
    """
    def __init__(self, model_path, gpu_id):
        """
        Initialize the YOLO model on a specific GPU.
        
        Args:
            model_path: Path to YOLO model file
            gpu_id: GPU device ID to use
        """
        self.gpu_id = gpu_id
        # Set the device for this actor
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Load model on GPU
        self.model = YOLO(model_path)
        self.model.to(f'cuda:0')  # Since CUDA_VISIBLE_DEVICES is set, use cuda:0
        
        print(f"[GPU {gpu_id}] Model loaded successfully")
    
    def detect_faces_batch(self, image_batch):
        """
        Detect faces in a batch of images using GPU.
        
        Args:
            image_batch: List of image paths
            
        Returns:
            List of dicts with detection results
        """
        data_list = []
        
        # Run YOLO detection on the batch with GPU
        results = self.model(image_batch, verbose=False, device='cuda:0')
        
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


def detect_faces_ray_gpu(imgs, model_path, batch_size, gpu_ids):
    """
    Detect faces using Ray for parallel GPU processing.
    
    Args:
        imgs: List of image paths
        model_path: Path to YOLO model file
        batch_size: Number of images to process per batch
        gpu_ids: List of GPU IDs to use (e.g., [0, 1, 2, 3] or [1, 4, 5, 6])
        
    Returns:
        List of detection dictionaries
    """
    # Initialize Ray with GPU resources
    if not ray.is_initialized():
        # Tell Ray about available GPUs
        ray.init(num_gpus=len(gpu_ids))
    
    print(f"Ray initialized with {len(gpu_ids)} GPUs: {gpu_ids}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Total CUDA devices detected: {torch.cuda.device_count()}")
    
    # Create Ray actors (one per GPU)
    print(f"Creating {len(gpu_ids)} Ray actors (one per GPU)...")
    actors = []
    for gpu_id in gpu_ids:
        actor = GPUFaceDetector.remote(model_path, gpu_id)
        actors.append(actor)
        print(f"  - Actor assigned to GPU {gpu_id}")
    
    # Split images into batches
    image_batches = []
    for i in range(0, len(imgs), batch_size):
        batch = imgs[i:i+batch_size]
        image_batches.append(batch)
    
    print(f"\nSplit {len(imgs)} images into {len(image_batches)} batches of size {batch_size}")
    print(f"Submitting tasks to GPU workers...")
    
    # Submit batches to actors in round-robin fashion
    futures = []
    for i, batch in enumerate(image_batches):
        actor = actors[i % len(actors)]
        future = actor.detect_faces_batch.remote(batch)
        futures.append(future)
    
    print(f"Distributed {len(futures)} batches across {len(actors)} GPUs")
    
    # Collect results with progress bar
    print(f"\nProcessing {len(futures)} batches in parallel on GPUs...")
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


def parse_gpu_ids(gpu_string):
    """
    Parse GPU ID string into list of integers.
    
    Args:
        gpu_string: Comma-separated GPU IDs (e.g., "0,1,2,3" or "1,4,5,6")
        
    Returns:
        List of GPU IDs as integers
    """
    return [int(x.strip()) for x in gpu_string.split(',')]


def main(input_dir, output_csv, batch_size, model_path, gpu_ids):
    """
    Main function to run face detection with Ray on GPUs.
    
    Args:
        input_dir: Directory containing images
        output_csv: Output CSV file path
        batch_size: Number of images to process per batch
        model_path: Path to YOLO model file
        gpu_ids: List of GPU IDs to use (e.g., [0, 1, 2, 3])
    """
    print(f"{'='*60}")
    print(f"RAY GPU-ACCELERATED FACE DETECTION")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  Input directory: {input_dir}")
    print(f"  Output CSV: {output_csv}")
    print(f"  Model: {model_path}")
    print(f"  Batch size: {batch_size}")
    print(f"  GPU IDs: {gpu_ids}")
    print(f"  Number of GPUs: {len(gpu_ids)}")
    print(f"{'='*60}\n")
    
    # Verify CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available! This script requires GPU support.")
        print("Please check your PyTorch and CUDA installation.")
        return
    
    # Get all images
    print(f"Scanning directory: {input_dir}")
    imgs = get_image_paths(input_dir)
    print(f"Total images found: {len(imgs)}")
    
    if len(imgs) == 0:
        print("No images found! Exiting.")
        return
    
    # Detect faces with Ray on GPUs
    print(f"\nStarting face detection with Ray on {len(gpu_ids)} GPUs...")
    detections = detect_faces_ray_gpu(imgs, model_path, batch_size, gpu_ids)
    
    # Save results
    print(f"\nSaving results...")
    total_faces = save_results(detections, output_csv)
    
    print(f"\n{'='*60}")
    print(f"Detection complete!")
    print(f"Total faces detected: {total_faces}")
    print(f"Total images processed: {len(imgs)}")
    if len(imgs) > 0:
        print(f"Average faces per image: {total_faces / len(imgs):.2f}")
    print(f"Results saved to: {output_csv}")
    print(f"{'='*60}")
    
    # Shutdown Ray
    ray.shutdown()


if __name__ == "__main__":
    # ========== CONFIGURATION ==========
    # Edit these variables as needed
    
    # Input/Output paths
    input_dir = '/data0/PArun/Quality_checklist/dataset/cropped_detected_shots/Sep/Devdas'
    output_csv = '/data0/teja_works/dataset_captioning/faster_actor_tagging/output/devadas_faces.csv'

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    # Model configuration
    model_path = 'yolov12m-face.pt'
    batch_size = 64  # Number of images per batch (increase for better GPU utilization)
    
    # GPU configuration
    # Option 1: Specify GPU IDs as a string (recommended for easy editing)
    gpu_string = "0,1,2,3,4,7"  # Use GPUs 1, 4, 5, and 6
    # gpu_string = "0,1,2,3,4,5,6,7"  # Use all 8 GPUs
    # gpu_string = "0"  # Use only GPU 0
    
    # Option 2: Or directly as a list
    # gpu_ids = [1, 4, 5, 6]
    
    # Parse GPU string to list
    gpu_ids = parse_gpu_ids(gpu_string)
    
    print(f"Configured to use GPUs: {gpu_ids}")
    print(f"Total GPUs to be used: {len(gpu_ids)}\n")
    
    # ===================================
    
    # Run face detection
    main(input_dir, output_csv, batch_size, model_path, gpu_ids)

