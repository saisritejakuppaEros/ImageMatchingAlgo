"""
Face Embedding Generation with Ray GPU Parallel Processing

This script uses Ray to parallelize face embedding generation across multiple GPUs,
significantly speeding up the processing of large datasets.

GPU CONFIGURATION:
------------------
- Specify which GPUs to use via gpu_ids (e.g., [0, 1, 2, 3] or [1, 4, 5, 6])
- Each GPU gets its own Ray actor with a dedicated DeepFace model
- Batches are distributed evenly across all specified GPUs

MAPPING STRUCTURE:
------------------
1. CSV Structure (e.g., face_detection.csv):
   - Columns: image_path, x1, y1, x2, y2, confidence, face_id
   - Each row = one detected face with a unique face_id
   - Multiple rows can have the same image_path (multiple faces in one image)

2. Pickle File Structure:
   - Filename: <original_image_name>_face_<face_id>.pkl
   - Location: Mirrors the input folder hierarchy in embeddings folder
   - Content: {face_id, embedding, model, bbox, confidence, image_path}

3. Example:
   CSV rows:
     - image_path=/frames/shot_1/0.jpg, bbox=(10,20,50,80), face_id=0
     - image_path=/frames/shot_1/0.jpg, bbox=(100,30,150,90), face_id=1
   
   Pickle files:
     - /embeddings/shot_1/0_face_0.pkl  (contains face_id=0, embedding, bbox, etc.)
     - /embeddings/shot_1/0_face_1.pkl  (contains face_id=1, embedding, bbox, etc.)

4. How to map back:
   - Read CSV and filter by face_id to get bbox and image_path
   - Use face_id to construct pickle filename
   - Load pickle to get embedding vector
"""

import os
import pickle
from pathlib import Path
import pandas as pd
import numpy as np
from PIL import Image
from tqdm import tqdm
import ray
from deepface import DeepFace
import torch


def get_output_path(image_path, input_root, output_root, face_id):
    """
    Convert input image path to output embedding path with unique face_id.
    Replaces the input_root with output_root and adds face_id to filename.
    
    Example:
        /path/to/frames/shot_001/1/frame.jpg, face_id=5 -> /path/to/embeddings/shot_001/1/frame_face_5.pkl
    """
    # Convert to Path objects
    img_path = Path(image_path)
    input_root_path = Path(input_root)
    output_root_path = Path(output_root)
    
    # Get the relative path from input_root
    try:
        relative_path = img_path.relative_to(input_root_path)
    except ValueError:
        # If image_path is not relative to input_root, just use the filename
        relative_path = Path(img_path.name)
    
    # Add face_id to filename before extension
    stem = relative_path.stem  # filename without extension
    new_filename = f"{stem}_face_{face_id}.pkl"
    relative_path = relative_path.parent / new_filename
    
    # Combine with output_root
    output_path = output_root_path / relative_path
    
    return output_path


@ray.remote(num_gpus=1)
class GPUEmbeddingGenerator:
    """
    Ray Actor that processes face embeddings on a specific GPU.
    Each actor is assigned to one GPU using Ray's GPU resource management.
    """
    def __init__(self, gpu_id, model_name='ArcFace'):
        """
        Initialize the embedding generator on a specific GPU.
        
        Args:
            gpu_id: GPU device ID to use
            model_name: DeepFace model to use for embeddings
        """
        self.gpu_id = gpu_id
        self.model_name = model_name
        
        # Set the device for this actor
        os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)
        
        # Force DeepFace to use GPU by warming it up
        try:
            # Create a dummy image to initialize the model on GPU
            dummy_img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
            DeepFace.represent(
                img_path=dummy_img,
                model_name=model_name,
                enforce_detection=False
            )
            print(f"[GPU {gpu_id}] DeepFace {model_name} model loaded successfully")
        except Exception as e:
            print(f"[GPU {gpu_id}] Warning during model initialization: {str(e)}")
    
    def process_face_batch(self, batch_data, input_root, output_root):
        """
        Process a batch of faces and generate embeddings on GPU.
        
        Args:
            batch_data: List of dicts with face detection info
            input_root: Root directory of input images
            output_root: Root directory for output embeddings
            
        Returns:
            dict with counts: {saved: int, skipped: int, errors: int}
        """
        embeddings_saved = 0
        errors = 0
        skipped = 0
        
        # Prepare batch images and metadata
        batch_images = []
        batch_metadata = []
        
        for face_data in batch_data:
            try:
                img_path = face_data['image_path']
                x1, y1, x2, y2 = face_data['x1'], face_data['y1'], face_data['x2'], face_data['y2']
                face_id = face_data['face_id']
                
                # Check if output already exists (skip if exists)
                output_path = get_output_path(img_path, input_root, output_root, face_id)
                if output_path.exists():
                    skipped += 1
                    continue
                
                # Load and crop image
                img = Image.open(img_path)
                img_cropped = img.crop((x1, y1, x2, y2))
                img_array = np.array(img_cropped)
                
                batch_images.append(img_array)
                batch_metadata.append({
                    'face_id': face_id,
                    'image_path': img_path,
                    'bbox': {'x1': x1, 'y1': y1, 'x2': x2, 'y2': y2},
                    'confidence': face_data['confidence'],
                    'output_path': output_path
                })
                
            except Exception as e:
                errors += 1
                print(f"[GPU {self.gpu_id}] Error loading {face_data['image_path']}: {str(e)}")
                continue
        
        # Skip if no images in batch
        if len(batch_images) == 0:
            return {'saved': embeddings_saved, 'skipped': skipped, 'errors': errors}
        
        # Generate embeddings for batch on GPU
        try:
            results = DeepFace.represent(
                img_path=batch_images,
                model_name=self.model_name,
                enforce_detection=False
            )
            
            # Save each embedding
            for i, (result, metadata) in enumerate(zip(results, batch_metadata)):
                try:
                    if isinstance(result, list) and len(result) > 0:
                        embedding_data = {
                            'face_id': metadata['face_id'],
                            'embedding': result[0]['embedding'],
                            'model': self.model_name,
                            'bbox': metadata['bbox'],
                            'confidence': metadata['confidence'],
                            'image_path': metadata['image_path']
                        }
                    else:
                        embedding_data = {
                            'face_id': metadata['face_id'],
                            'embedding': result['embedding'],
                            'model': self.model_name,
                            'bbox': metadata['bbox'],
                            'confidence': metadata['confidence'],
                            'image_path': metadata['image_path']
                        }
                    
                    # Create output directory if it doesn't exist
                    metadata['output_path'].parent.mkdir(parents=True, exist_ok=True)
                    
                    # Save embedding as pickle
                    with open(metadata['output_path'], 'wb') as f:
                        pickle.dump(embedding_data, f)
                    
                    embeddings_saved += 1
                    
                except Exception as e:
                    errors += 1
                    print(f"[GPU {self.gpu_id}] Error saving embedding for {metadata['image_path']}: {str(e)}")
                    continue
                    
        except Exception as e:
            errors += len(batch_images)
            print(f"[GPU {self.gpu_id}] Error processing batch: {str(e)}")
        
        return {'saved': embeddings_saved, 'skipped': skipped, 'errors': errors}


def generate_embeddings_ray_gpu(csv_path, input_root, output_root, model_name='ArcFace', 
                                 batch_size=32, gpu_ids=[0]):
    """
    Generate face embeddings using Ray for parallel GPU processing.
    
    Args:
        csv_path: Path to CSV file with face detections (columns: image_path, x1, y1, x2, y2, confidence)
        input_root: Root directory of input images (e.g., 'frames' or 'actors')
        output_root: Root directory for output embeddings (e.g., 'embeddings')
        model_name: DeepFace model to use for embeddings
        batch_size: Number of faces to process in each batch
        gpu_ids: List of GPU IDs to use (e.g., [0, 1, 2, 3])
    """
    # Initialize Ray with GPU resources
    if not ray.is_initialized():
        ray.init(num_gpus=len(gpu_ids))
    
    print(f"Ray initialized with {len(gpu_ids)} GPUs: {gpu_ids}")
    print(f"CUDA available: {torch.cuda.is_available()}")
    print(f"Total CUDA devices detected: {torch.cuda.device_count()}")
    
    # Read CSV
    print(f"\nReading CSV from: {csv_path}")
    df = pd.read_csv(csv_path)
    
    # Add face_id column if it doesn't exist (unique ID for each face detection)
    if 'face_id' not in df.columns:
        df['face_id'] = range(len(df))
        # Save updated CSV with face_id
        df.to_csv(csv_path, index=False)
        print(f"Added face_id column to CSV")
    
    print(f"Found {len(df)} face detections")
    
    # Convert DataFrame rows to list of dicts for Ray
    face_data_list = df.to_dict('records')
    
    # Create Ray actors (one per GPU)
    print(f"\nCreating {len(gpu_ids)} Ray actors (one per GPU)...")
    actors = []
    for gpu_id in gpu_ids:
        actor = GPUEmbeddingGenerator.remote(gpu_id, model_name)
        actors.append(actor)
        print(f"  - Actor assigned to GPU {gpu_id}")
    
    # Split into batches
    batches = []
    for i in range(0, len(face_data_list), batch_size):
        batch = face_data_list[i:i + batch_size]
        batches.append(batch)
    
    print(f"\nSplit into {len(batches)} batches of size {batch_size}")
    print(f"Submitting tasks to GPU workers...")
    
    # Submit all batches to Ray actors in round-robin fashion
    futures = []
    for i, batch in enumerate(batches):
        actor = actors[i % len(actors)]
        future = actor.process_face_batch.remote(batch, input_root, output_root)
        futures.append(future)
    
    print(f"Distributed {len(futures)} batches across {len(actors)} GPUs")
    
    # Collect results with progress bar
    print(f"\nProcessing {len(futures)} batches in parallel on GPUs...")
    results = []
    for future in tqdm(futures, desc="Processing batches"):
        result = ray.get(future)
        results.append(result)
    
    # Aggregate results
    total_saved = sum(r['saved'] for r in results)
    total_skipped = sum(r['skipped'] for r in results)
    total_errors = sum(r['errors'] for r in results)
    
    print(f"\n{'='*60}")
    print(f"Embedding generation complete!")
    print(f"Successfully saved: {total_saved} embeddings")
    print(f"Skipped (already exists): {total_skipped}")
    print(f"Errors: {total_errors}")
    print(f"Output directory: {output_root}")
    print(f"{'='*60}")
    
    # Shutdown Ray
    ray.shutdown()


def load_embedding(pkl_path):
    """
    Load an embedding from a pickle file.
    
    Returns:
        dict with keys: face_id, embedding, model, bbox, confidence, image_path
    """
    with open(pkl_path, 'rb') as f:
        return pickle.load(f)


def get_embeddings_for_image(csv_path, image_path, embeddings_root):
    """
    Get all embeddings for a specific image.
    
    Args:
        csv_path: Path to CSV file with face detections
        image_path: Path to the image
        embeddings_root: Root directory of embeddings
        
    Returns:
        List of tuples: [(face_id, embedding_pkl_path, bbox), ...]
    """
    df = pd.read_csv(csv_path)
    faces = df[df['image_path'] == image_path]
    
    results = []
    for _, row in faces.iterrows():
        face_id = int(row['face_id'])
        results.append({
            'face_id': face_id,
            'bbox': {'x1': row['x1'], 'y1': row['y1'], 'x2': row['x2'], 'y2': row['y2']},
            'confidence': row['confidence']
        })
    
    return results


def parse_gpu_ids(gpu_string):
    """
    Parse GPU ID string into list of integers.
    
    Args:
        gpu_string: Comma-separated GPU IDs (e.g., "0,1,2,3" or "1,4,5,6")
        
    Returns:
        List of GPU IDs as integers
    """
    return [int(x.strip()) for x in gpu_string.split(',')]


def main(csv_path, input_root, output_root, model_name, batch_size, gpu_ids):
    """Main function to generate embeddings with Ray on GPUs."""
    print(f"{'='*60}")
    print(f"RAY GPU-ACCELERATED EMBEDDING GENERATION")
    print(f"{'='*60}")
    print(f"Configuration:")
    print(f"  CSV path: {csv_path}")
    print(f"  Input root: {input_root}")
    print(f"  Output root: {output_root}")
    print(f"  Model: {model_name}")
    print(f"  Batch size: {batch_size}")
    print(f"  GPU IDs: {gpu_ids}")
    print(f"  Number of GPUs: {len(gpu_ids)}")
    print(f"{'='*60}\n")
    
    # Verify CUDA availability
    if not torch.cuda.is_available():
        print("WARNING: CUDA is not available! This script requires GPU support.")
        print("Please check your PyTorch and CUDA installation.")
        return
    
    generate_embeddings_ray_gpu(csv_path, input_root, output_root, model_name, batch_size, gpu_ids)
    
    print(f"\n{'='*60}")
    print("MAPPING STRUCTURE:")
    print("  CSV Row <-> Pickle File Mapping:")
    print("    - Each row in CSV has a unique 'face_id'")
    print("    - Pickle filename: <original_image>_face_<face_id>.pkl")
    print("    - Pickle contains: face_id, embedding, bbox, confidence, image_path")
    print(f"  Example:")
    print(f"    CSV row with face_id=5, image=/path/frames/shot_1/1.jpg")
    print(f"    -> Saved as: /path/embeddings/shot_1/1_face_5.pkl")
    print(f"{'='*60}")


if __name__ == "__main__":
    # ========== CONFIGURATION ==========
    # Edit these variables as needed
    
    # Input/Output paths
    csv_path = 'output/devadas_faces.csv'
    input_root = '/data0/PArun/Quality_checklist/dataset/cropped_detected_shots/Sep/Devdas'
    output_root = 'output/devdas_embeddings'
    
    # Model configuration
    model_name = 'ArcFace'  # Options: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, ArcFace, Dlib, SFace, GhostFaceNet, Buffalo_L
    batch_size = 64  # Number of faces to process per batch (increase for better GPU utilization)
    
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
    
    # Run embedding generation
    main(csv_path, input_root, output_root, model_name, batch_size, gpu_ids)

