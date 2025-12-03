#!/usr/bin/env python3
"""
FiftyOne visualization of face detections with actor similarity scores
"""

import pandas as pd
import fiftyone as fo
from PIL import Image


def create_visualization_dataset(csv_path):
    """
    Create FiftyOne dataset from actor search results CSV
    
    Args:
        csv_path: Path to actor_search_results.csv
    """
    # Load CSV data
    df = pd.read_csv(csv_path)
    
    # Create FiftyOne dataset
    dataset_name = "actor_face_matches"
    
    # Delete existing dataset if present
    if dataset_name in fo.list_datasets():
        fo.delete_dataset(dataset_name)
    
    dataset = fo.Dataset(name=dataset_name)
    
    # Group by image path
    samples = []
    for image_path, group in df.groupby('frame_image_path'):
        # Get image dimensions
        try:
            img = Image.open(image_path)
            w, h = img.size
        except Exception as e:
            print(f"Skipping {image_path}: {e}")
            continue
        
        # Create sample
        sample = fo.Sample(filepath=image_path)
        
        # Add detections for this image
        detections = []
        for _, row in group.iterrows():
            # Convert absolute coords to relative [0, 1]
            x1, y1, x2, y2 = row['face_x1'], row['face_y1'], row['face_x2'], row['face_y2']
            rel_box = [x1 / w, y1 / h, (x2 - x1) / w, (y2 - y1) / h]
            
            # Create detection with actor name and similarity score
            label = f"{row['actor_name']} ({row['similarity_score']:.3f})"
            
            detections.append(
                fo.Detection(
                    label=label,
                    bounding_box=rel_box,
                    confidence=row['similarity_score']
                )
            )
        
        sample["faces"] = fo.Detections(detections=detections)
        samples.append(sample)
    
    # Add samples to dataset
    dataset.add_samples(samples)
    
    print(f"\nDataset created: {len(samples)} images with face detections")
    print(f"Total matches: {len(df)}")
    
    return dataset


if __name__ == "__main__":
    csv_path = "output/fiass/actor_search_results.csv"
    
    # Create dataset
    dataset = create_visualization_dataset(csv_path)
    
    # Launch FiftyOne App
    session = fo.launch_app(dataset)
    
    print("\nFiftyOne App launched!")
    print("- Face detections shown with actor names and similarity scores")
    print("- Use confidence slider in 'faces' field to filter by similarity score")
    print("- Click images to view in detail")
    
    # Keep session alive
    session.wait()

