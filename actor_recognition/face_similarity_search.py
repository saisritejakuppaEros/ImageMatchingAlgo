import os
import json
import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.spatial.distance import cosine
import csv
from pathlib import Path
import shutil
from tqdm import tqdm

def load_face_data(embedding_file):
    """Load all face embeddings from a file"""
    with open(embedding_file, 'r') as f:
        data = json.load(f)
        faces = []
        for face_key, face_data in data.items():
            if 'embedding' in face_data:
                faces.append({
                    'face_key': face_key,
                    'embedding': np.array(face_data['embedding']),
                    'bbox': face_data['yolo_detection']['bbox'] if 'yolo_detection' in face_data else None,
                    'original_image': face_data['original_image'],
                    'source_file': embedding_file
                })
        return faces

def calculate_cosine_similarity(emb1, emb2):
    return 1 - cosine(emb1, emb2)

def draw_bbox_and_save(image_path, matches, output_path):
    img = Image.open(image_path)
    draw = ImageDraw.Draw(img)
    
    # Use different colors for different similarity ranges
    def get_color(similarity):
        if similarity >= 0.8: return 'green'
        elif similarity >= 0.6: return 'yellow'
        elif similarity >= 0.4: return 'orange'
        else: return 'red'
    
    # Draw boxes and scores for each match
    for match in matches:
        bbox = match['bbox']
        similarity = match['similarity']
        color = get_color(similarity)
        
        # Draw rectangle
        x1, y1 = bbox['x1'], bbox['y1']
        x2, y2 = bbox['x2'], bbox['y2']
        draw.rectangle([x1, y1, x2, y2], outline=color, width=3)
        
        # Draw similarity score and match info
        score_text = f"Match: {similarity:.3f}"
        draw.text((x1, y1-20), score_text, fill=color)
    
    img.save(output_path)

def main():
    print("Starting face similarity search...")
    
    # Create output directory
    output_dir = "embed_search"
    os.makedirs(output_dir, exist_ok=True)
    
    # Initialize CSV file
    csv_path = os.path.join(output_dir, "similarity_scores.csv")
    with open(csv_path, 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow([
            'Source Image',
            'Source Face',
            'Target Image',
            'Target Face',
            'Similarity',
            'Source File',
            'Target File'
        ])
    
    # Get list of all embedding files
    processed_images_dir = "/data0/teja_codes/actor_tagging/processed_images"
    embedding_files = []
    for root, dirs, files in os.walk(processed_images_dir):
        for file in files:
            if file.endswith('_embeddings.json'):
                embedding_files.append(os.path.join(root, file))
    
    print(f"Found {len(embedding_files)} embedding files to process")
    
    # Load all face data first
    all_faces = []
    for embedding_file in tqdm(embedding_files, desc="Loading face data", unit="file"):
        faces = load_face_data(embedding_file)
        all_faces.extend(faces)
    
    print(f"Loaded {len(all_faces)} total faces")
    
    # Process each face against all others
    processed_images = set()
    similarity_threshold = 0.5  # Only save matches above this threshold
    
    for i, face1 in enumerate(tqdm(all_faces, desc="Processing faces", unit="face")):
        matches_for_image = []
        
        # Compare with all other faces
        for j, face2 in enumerate(all_faces):
            if i != j:  # Don't compare face with itself
                similarity = calculate_cosine_similarity(face1['embedding'], face2['embedding'])
                
                if similarity >= similarity_threshold:
                    # Save to CSV
                    with open(csv_path, 'a', newline='') as csvfile:
                        csvwriter = csv.writer(csvfile)
                        csvwriter.writerow([
                            face1['original_image'],
                            face1['face_key'],
                            face2['original_image'],
                            face2['face_key'],
                            similarity,
                            face1['source_file'],
                            face2['source_file']
                        ])
                    
                    # Add to matches for visualization
                    if face1['original_image'] not in processed_images:
                        matches_for_image.append({
                            'bbox': face1['bbox'],
                            'similarity': similarity
                        })
        
        # Save visualization if we haven't processed this image yet
        if face1['original_image'] not in processed_images and matches_for_image:
            # Create output image name
            image_name = os.path.basename(face1['original_image'])
            output_image_name = f"matches_{image_name}"
            output_path = os.path.join(output_dir, output_image_name)
            
            # Draw bboxes and save image
            draw_bbox_and_save(face1['original_image'], matches_for_image, output_path)
            processed_images.add(face1['original_image'])
    
    print(f"\nProcessing complete! Results saved in {output_dir}/")
    print(f"CSV log file: {csv_path}")
    print(f"Processed {len(processed_images)} unique images")

if __name__ == "__main__":
    main() 