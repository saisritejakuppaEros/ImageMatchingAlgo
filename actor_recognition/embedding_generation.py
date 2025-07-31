from ultralytics import YOLO
import pandas as pd
import cv2
import os
import json
from deepface import DeepFace
from pathlib import Path
import shutil
import numpy as np
import warnings
import random

# Configuration variables
INPUT_DIR = "/data0/teja_codes/dataset_generation_pipeline/MovieDatasetGeneration/Pipeline/output/images"
OUTPUT_DIR = "./processed_images"


# INPUT_DIR = "/data0/teja_codes/ImmersoAiResearch/ImageMatchingAlgo/actor_recognition/actor_images"
# OUTPUT_DIR = "./processed_images_actor"


NUM_IMAGES = None  # Set to None to process all images
RANDOM_SEED = 42   # For reproducible random sampling

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    # Place yolob/YOLOv5 calls that trigger deprecation warnings here

from tqdm import tqdm

def ensure_dir(dir_path):
    """Create directory if it doesn't exist"""
    Path(dir_path).mkdir(parents=True, exist_ok=True)

def crop_face(img, x1, y1, x2, y2, padding=10):
    """Crop face with padding and handle boundary cases"""
    height, width = img.shape[:2]
    
    # Add padding but ensure we don't go out of bounds
    x1_pad = max(0, x1 - padding)
    y1_pad = max(0, y1 - padding)
    x2_pad = min(width, x2 + padding)
    y2_pad = min(height, y2 + padding)
    
    return img[y1_pad:y2_pad, x1_pad:x2_pad]

def process_image(img_path, yolo_model, output_base_dir):
    """Process a single image with face detection and embedding generation"""
    # Create relative path structure
    rel_path = os.path.relpath(img_path, input_base_dir)
    output_dir = os.path.join(output_base_dir, os.path.dirname(rel_path))
    ensure_dir(output_dir)
    
    # Base filename without extension
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    
    # Create faces directory for this image
    faces_dir = os.path.join(output_dir, f"{base_name}_faces")
    ensure_dir(faces_dir)
    
    # Output paths
    csv_path = os.path.join(output_dir, f"{base_name}_faces.csv")
    json_path = os.path.join(output_dir, f"{base_name}_embeddings.json")
    vis_path = os.path.join(output_dir, f"{base_name}_detected.jpg")
    
    # Step 1: YOLO Face Detection
    results = yolo_model.predict(
        source=img_path,
        conf=0.25,
        imgsz=1280,
        line_width=1,
        max_det=1000,
        verbose=False
    )
    
    # Initialize DataFrame for detections
    df = pd.DataFrame()
    
    # Read the original image
    img = cv2.imread(img_path)
    if img is None:
        print(f"Error: Could not read image {img_path}")
        return 0, 0
    
    # Dictionary to store face embeddings
    all_embeddings = {}
    
    # Process YOLO results
    face_count = 0
    for r in results:
        if r.boxes is not None:
            boxes_df = pd.DataFrame(
                r.boxes.data.cpu().numpy(),
                columns=['x1', 'y1', 'x2', 'y2', 'confidence', 'class']
            )
            df = pd.concat([df, boxes_df], ignore_index=True)
            
            # Process each detected face with progress bar
            with tqdm(total=len(boxes_df), desc="Processing Faces", unit="face", leave=False) as face_pbar:
                for idx, row in boxes_df.iterrows():
                    x1, y1, x2, y2 = map(int, [row['x1'], row['y1'], row['x2'], row['y2']])
                    confidence = row['confidence']
                    
                    # Crop face with padding
                    face_img = crop_face(img, x1, y1, x2, y2)
                    
                    # Save cropped face
                    face_path = os.path.join(faces_dir, f"face_{face_count}.jpg")
                    cv2.imwrite(face_path, face_img)
                    
                    # Generate embedding for cropped face
                    try:
                        embedding = DeepFace.represent(img_path=face_path, model_name="ArcFace", enforce_detection=False)
                        all_embeddings[f"face_{face_count}"] = {
                            "embedding": embedding[0]["embedding"] if embedding else None,
                            "yolo_detection": {
                                "bbox": {
                                    "x1": int(x1),
                                    "y1": int(y1),
                                    "x2": int(x2),
                                    "y2": int(y2)
                                },
                                "confidence": float(confidence),
                                "width": int(x2 - x1),
                                "height": int(y2 - y1),
                                "center_x": int((x1 + x2) / 2),
                                "center_y": int((y1 + y2) / 2),
                                "area": int((x2 - x1) * (y2 - y1))
                            },
                            "face_path": face_path,  # Path to the cropped face
                            "original_image": img_path  # Original image path
                        }
                    except Exception as e:
                        print(f"\nError generating embedding for face {face_count} in {img_path}: {str(e)}")
                    
                    # Draw on visualization image
                    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
                    text = f"Face {face_count} (Conf: {confidence:.2f})"
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    font_scale = 0.6
                    thickness = 2
                    
                    (text_width, text_height), _ = cv2.getTextSize(text, font, font_scale, thickness)
                    cv2.rectangle(img, (x1, y1-text_height-10), (x1+text_width+10, y1), (0, 255, 0), -1)
                    cv2.putText(img, text, (x1+5, y1-5), font, font_scale, (0, 0, 0), thickness)
                    
                    face_count += 1
                    face_pbar.update(1)
                    face_pbar.set_postfix({"Current": f"Face {face_count}"})
    
    # Save detections to CSV
    df.to_csv(csv_path, index=False)
    
    # Save embeddings to JSON
    # Convert numpy arrays to lists for JSON serialization
    for face_id in all_embeddings:
        if all_embeddings[face_id]["embedding"] is not None:
            # Check if embedding is numpy array before converting to list
            if isinstance(all_embeddings[face_id]["embedding"], np.ndarray):
                all_embeddings[face_id]["embedding"] = all_embeddings[face_id]["embedding"].tolist()
            # If it's already a list, keep it as is
    
    with open(json_path, 'w') as f:
        json.dump(all_embeddings, f, indent=4)
    
    # Save visualization
    cv2.imwrite(vis_path, img)
    
    return face_count, len(all_embeddings)

def get_all_images(base_dir):
    """Get all image paths from the directory recursively"""
    image_paths = []
    for root, _, files in os.walk(base_dir):
        for file in sorted(files):
            if file.lower().endswith(('.png', '.jpg', '.jpeg')):
                image_paths.append(os.path.join(root, file))
    return image_paths

if __name__ == "__main__":
    # Initialize YOLO model
    model = YOLO("yolov12m-face.pt")
    
    # Input and output directories
    input_base_dir = INPUT_DIR
    output_base_dir = OUTPUT_DIR
    
    # Ensure output base directory exists
    ensure_dir(output_base_dir)
    
    # Get all image paths
    all_image_paths = get_all_images(input_base_dir)
    import random
    random.seed(RANDOM_SEED)
    # shuggle the images
    random.shuffle(all_image_paths)
    total_available = len(all_image_paths)
    
    # If num_images is specified, randomly sample that many images
    if NUM_IMAGES is not None:
        if NUM_IMAGES > total_available:
            print(f"Warning: Requested {NUM_IMAGES} images but only {total_available} are available.")
            num_images = total_available
        else:
            num_images = NUM_IMAGES
        
        # Set random seed for reproducibility
        random.seed(RANDOM_SEED)
        selected_images = random.sample(all_image_paths, num_images)
    else:
        selected_images = all_image_paths
        num_images = total_available
    
    print(f"\nProcessing {num_images} images out of {total_available} total images")
    print(f"Output directory: {output_base_dir}")
    
    # Process selected images with progress bar
    total_images = 0
    total_faces = 0
    total_embeddings = 0
    
    # Save the list of selected images
    selected_images_file = os.path.join(output_base_dir, "processed_images_list.txt")
    with open(selected_images_file, 'w') as f:
        for img_path in selected_images:
            f.write(f"{img_path}\n")
    
    # Main progress bar for selected images
    with tqdm(total=num_images, desc="Processing Images", unit="img") as pbar:
        for img_path in selected_images:
            rel_path = os.path.relpath(img_path, input_base_dir)
            pbar.set_postfix({
                "Current": rel_path, 
                "Faces": total_faces,
                "Embeddings": total_embeddings
            })
            
            try:
                faces, embs = process_image(img_path, model, output_base_dir)
                total_images += 1
                total_faces += faces
                total_embeddings += embs
                pbar.set_postfix({
                    "Current": rel_path, 
                    "Faces": total_faces,
                    "Embeddings": total_embeddings
                })
            except Exception as e:
                print(f"\nError processing {img_path}: {str(e)}")
            
            pbar.update(1)
    
    # Save processing summary
    summary = {
        "total_images_available": total_available,
        "images_processed": total_images,
        "total_faces_detected": total_faces,
        "total_embeddings_generated": total_embeddings,
        "random_seed_used": RANDOM_SEED if NUM_IMAGES is not None else None,
        "sampling_rate": f"{total_images}/{total_available}" if NUM_IMAGES is not None else "all"
    }
    
    with open(os.path.join(output_base_dir, "processing_summary.json"), 'w') as f:
        json.dump(summary, f, indent=4)
    
    print(f"\nProcessing complete!")
    print(f"Total images processed: {total_images}")
    print(f"Total faces detected: {total_faces}")
    print(f"Total embeddings generated: {total_embeddings}")
    print(f"Processing summary saved to: {os.path.join(output_base_dir, 'processing_summary.json')}")
    print(f"List of processed images saved to: {selected_images_file}")
