"""
Actor Face Visualization Script
================================

This script visualizes actor face detection results by drawing bounding boxes
and labels on images with multiple faces per image support.

Features:
- Groups faces by image (handles multiple faces per image correctly)
- Draws labeled bounding boxes with actor names and similarity scores
- Supports filtering by similarity threshold
- Maintains original directory structure
"""

import os
import pandas as pd
from PIL import Image, ImageDraw, ImageFont
from tqdm import tqdm

# ========== CONFIGURATION ==========

og_images_dir = '/data0/PArun/Quality_checklist/dataset/cropped_detected_shots/Sep/Devdas'


# Input CSV path (from actor_search.py)
csv_path = '/data0/teja_works/dataset_captioning/faster_actor_tagging/output/faiss/results/actor_search_results_combined.csv'

# Output directory for annotated images
output_path = 'devadas'

# Similarity threshold (only draw faces with score >= threshold)
# Set to 0.0 to include all results, or higher (e.g., 0.5) to filter
similarity_threshold = 0.0

# Actor name mapping (for display purposes)
mapping = {
    'aish_face_0.pkl': 'Aishwarya Rai',
    'madhuri_face_1.pkl': 'Madhuri Dixit',
    'srk_face_2.pkl': 'Shah Rukh Khan'
}

# Drawing settings
box_color = 'red'
box_width = 3
text_color = 'white'
font_size = 16

# ===================================

if __name__ == "__main__":
    print("="*80)
    print("ACTOR FACE VISUALIZATION")
    print("="*80)
    print(f"CSV path: {csv_path}")
    print(f"Output path: {output_path}")
    print(f"Similarity threshold: {similarity_threshold}")
    print("="*80)

    # Load CSV
    df = pd.read_csv(csv_path)
    print(f"\nLoaded {len(df)} face detections from CSV")
    # Filter by similarity threshold
    if similarity_threshold > 0.0:
        df_filtered = df[df['similarity_score'] >= similarity_threshold]
        print(f"Filtered to {len(df_filtered)} faces with similarity >= {similarity_threshold}")
    else:
        df_filtered = df
        print("No filtering applied (threshold = 0.0)")

    # Create output directory
    os.makedirs(output_path, exist_ok=True)

    # Try to load a font for better text rendering, fallback to default if not available
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", font_size)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", font_size)
        except:
            font = ImageFont.load_default()
            print("Warning: Could not load TrueType font, using default font")

    # Group by frame_image_path to process each image only once
    print(f"\nProcessing images...")
    print(f"Total face detections: {len(df_filtered)}")
    print(f"Unique images to process: {df_filtered['frame_image_path'].nunique()}")

    # Actor statistics
    print(f"\nFaces per actor:")
    actor_counts = df_filtered['actor_name'].value_counts()
    for actor, count in actor_counts.items():
        display_name = mapping.get(actor + '.pkl', actor)
        print(f"  - {display_name}: {count} faces")

    grouped = df_filtered.groupby('frame_image_path')

    # Color palette for different actors
    actor_colors = {
        'aish_face_0': 'red',
        'madhuri_face_1': 'blue',
        'srk_face_2': 'green',
        'default': 'yellow'
    }

    print(f"\n{'='*80}")
    print("Starting image processing...")
    print(f"{'='*80}\n")

    images_processed = 0
    images_with_errors = 0

    for frame_image_path, group_df in tqdm(grouped, total=len(grouped), desc="Processing images"):
        try:
            # Load the image once
            image = Image.open(frame_image_path)
            draw = ImageDraw.Draw(image)
            
            # Draw all faces in this image
            for _, row in group_df.iterrows():
                face_x1 = int(row['face_x1'])
                face_y1 = int(row['face_y1'])
                face_x2 = int(row['face_x2'])
                face_y2 = int(row['face_y2'])
                actor_name = row['actor_name']
                similarity_score = row['similarity_score']
                
                # Use mapping to get full actor name if available
                actor_pkl_name = os.path.basename(row['actor_embedding_path'])
                display_name = mapping.get(actor_pkl_name, actor_name)
                
                # Get color for this actor
                color = actor_colors.get(actor_name, actor_colors['default'])
                
                # Draw rectangle around the face
                draw.rectangle((face_x1, face_y1, face_x2, face_y2), outline=color, width=box_width)
                
                # Create label with actor name and similarity score
                label = f"{display_name} ({similarity_score:.3f})"
                
                # Calculate text position (above the box if possible, otherwise inside)
                text_y = max(face_y1 - 25, 5)  # Position above box or at top of image
                
                # Draw text background for better visibility
                try:
                    bbox = draw.textbbox((face_x1, text_y), label, font=font)
                    # Add padding to the background
                    bbox = (bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2)
                    draw.rectangle(bbox, fill=color)
                except:
                    # Fallback if textbbox not available
                    draw.rectangle((face_x1, text_y, face_x1 + 200, text_y + 20), fill=color)
                
                # Write the label
                draw.text((face_x1, text_y), label, fill=text_color, font=font)
            
            # Save the image once with all faces drawn
            # Extract shot_xxxx/xx/filename.jpg hierarchy
            path_parts = frame_image_path.split('/')
            
            # Find the index where 'shot_' appears
            shot_idx = None
            for i, part in enumerate(path_parts):
                if part.startswith('shot_'):
                    shot_idx = i
                    break
            
            if shot_idx is not None:
                # Get everything from shot_xxxx onwards
                relative_path = '/'.join(path_parts[shot_idx:])
            else:
                # Fallback: use last 3 parts (shot_xxxx/xx/filename.jpg)
                relative_path = '/'.join(path_parts[-3:])
            
            save_path = os.path.join(output_path, relative_path)
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            image.save(save_path)
            images_processed += 1
            
        except Exception as e:
            print(f"\nError processing {frame_image_path}: {e}")
            images_with_errors += 1
            continue

    print(f"\n{'='*80}")
    print("PROCESSING COMPLETE!")
    print(f"{'='*80}")
    print(f"Images successfully processed: {images_processed}")
    print(f"Images with errors: {images_with_errors}")
    print(f"Total faces drawn: {len(df_filtered)}")
    print(f"Output directory: {output_path}")
    print(f"{'='*80}")