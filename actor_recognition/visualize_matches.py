import fiftyone as fo
import pandas as pd
import os
from collections import defaultdict

def create_fiftyone_dataset_from_csv(csv_path, dataset_name="face_detection_dataset"):
    """
    Create a FiftyOne dataset from CSV containing face detection data.
    
    Args:
        csv_path (str): Path to the CSV file
        dataset_name (str): Name for the FiftyOne dataset
    
    Returns:
        fo.Dataset: FiftyOne dataset with face detections
    """
    
    # Read the CSV file
    df = pd.read_csv(csv_path)
    
    # Group detections by image path (handling multiple detections per image)
    image_groups = defaultdict(list)
    
    print(f"Processing {len(df)} detection records...")
    
    for _, row in df.iterrows():
        image_path = row['original_image']
        
        # Store detection data for this image
        detection_data = {
            'face_id': row['face_id'],
            'similarity': row['similarity'],
            'shot': row['shot'],
            'frame': row['frame'],
            'x1': row['x1'],
            'y1': row['y1'],
            'x2': row['x2'],
            'y2': row['y2']
        }
        
        image_groups[image_path].append(detection_data)
    
    print(f"Found {len(image_groups)} unique images with detections")
    
    # Print statistics about multi-detection images
    multi_detection_images = {img: dets for img, dets in image_groups.items() if len(dets) > 1}
    if multi_detection_images:
        print(f"Images with multiple detections: {len(multi_detection_images)}")
        for img, dets in list(multi_detection_images.items())[:3]:  # Show first 3 examples
            print(f"  {os.path.basename(img)}: {len(dets)} detections")
    
    # Create FiftyOne samples
    samples = []
    
    for image_path, detections_data in image_groups.items():
        # Check if image file exists
        if not os.path.exists(image_path):
            print(f"Warning: Image not found - {image_path}")
            continue
            
        # Get image dimensions for normalization
        try:
            from PIL import Image
            with Image.open(image_path) as img:
                img_width, img_height = img.size
        except Exception as e:
            print(f"Error reading image {image_path}: {e}")
            continue
        
        # Create detection objects (multiple detections per image)
        detections = []
        
        print(f"Processing {len(detections_data)} detections for image: {os.path.basename(image_path)}")
        
        for i, det_data in enumerate(detections_data):
            # Convert absolute coordinates to normalized coordinates [0, 1]
            x1, y1, x2, y2 = det_data['x1'], det_data['y1'], det_data['x2'], det_data['y2']
            
            # Validate coordinates
            if x2 <= x1 or y2 <= y1:
                print(f"Warning: Invalid bounding box for detection {i} in {image_path}")
                continue
            
            # Normalize coordinates
            norm_x1 = max(0, x1 / img_width)
            norm_y1 = max(0, y1 / img_height)
            norm_width = min(1 - norm_x1, (x2 - x1) / img_width)
            norm_height = min(1 - norm_y1, (y2 - y1) / img_height)
            
            # Create FiftyOne detection with unique identifier
            detection = fo.Detection(
                label=det_data['face_id'],
                bounding_box=[norm_x1, norm_y1, norm_width, norm_height],
                confidence=det_data['similarity'],
                shot=det_data['shot'],
                frame=det_data['frame'],
                detection_id=f"{det_data['shot']}_{det_data['frame']}_{det_data['face_id']}"
            )
            
            detections.append(detection)
        
        # Create sample with all detections for this image
        sample = fo.Sample(
            filepath=image_path,
            face_detections=fo.Detections(detections=detections)
        )
        
        # Add metadata from first detection (since all detections in same image share shot/frame)
        if detections_data:
            sample['shot'] = detections_data[0]['shot']
            sample['frame'] = detections_data[0]['frame']
            sample['num_faces'] = len(detections)
            sample['avg_similarity'] = sum(d['similarity'] for d in detections_data) / len(detections_data)
        
        samples.append(sample)
    
    # Create dataset
    if fo.dataset_exists(dataset_name):
        dataset = fo.load_dataset(dataset_name)
        dataset.delete()
    
    dataset = fo.Dataset(dataset_name)
    dataset.add_samples(samples)
    
    return dataset

def load_and_explore_dataset(csv_path):
    """
    Load the dataset and provide basic exploration functionality.
    """
    
    # Create the dataset
    dataset = create_fiftyone_dataset_from_csv(csv_path)
    
    print(f"Dataset created with {len(dataset)} samples")
    print(f"Sample fields: {dataset.get_field_schema()}")
    
    # Print detailed statistics
    print(f"\nDataset statistics:")
    print(f"Total samples (unique images): {len(dataset)}")
    
    # Count total detections across all samples
    total_detections = sum(len(sample.face_detections.detections) for sample in dataset if sample.face_detections)
    print(f"Total face detections: {total_detections}")
    
    # Show examples of multi-detection images
    multi_face_samples = [s for s in dataset if s.num_faces > 1]
    if multi_face_samples:
        print(f"Images with multiple faces: {len(multi_face_samples)}")
        print("Examples:")
        for sample in multi_face_samples[:3]:
            face_ids = [det.label for det in sample.face_detections.detections]
            print(f"  {os.path.basename(sample.filepath)}: {sample.num_faces} faces {face_ids}")
    
    if len(dataset) > 0:
        sample = dataset.first()
        if sample.face_detections:
            print(f"\nFirst sample example:")
            print(f"Image: {os.path.basename(sample.filepath)}")
            print(f"Detections: {len(sample.face_detections.detections)}")
            for i, det in enumerate(sample.face_detections.detections):
                print(f"  Detection {i+1}: {det.label} (confidence: {det.confidence:.3f})")
    
    return dataset

# # Example usage
# if __name__ == "__main__":
#     csv_path = "/data0/teja_codes/actor_tagging/all_similarity_scores.csv"
    
#     # Load the dataset
#     dataset = load_and_explore_dataset(csv_path)
    
#     # Launch FiftyOne App to visualize
#     session = fo.launch_app(dataset)
    
#     # Keep the session alive
#     session.wait()

# Alternative: Quick loader function
def quick_load_face_dataset(csv_path, dataset_name="faces"):
    """
    Quick loader function that you can use directly.
    """
    return create_fiftyone_dataset_from_csv(csv_path, dataset_name)

# Usage examples:



from fiftyone import ViewField as F

# 1. Basic loading
dataset = quick_load_face_dataset("/data0/teja_codes/actor_tagging/all_similarity_scores.csv")

# 2. Filter by similarity threshold
high_confidence = dataset.filter_labels("face_detections", F("confidence") > 0.7)


# Launch FiftyOne App to visualize
session = fo.launch_app(high_confidence)

# Keep the session alive
session.wait()


# 3. Filter by specific face_id
# specific_face = dataset.filter_labels("face_detections", F("label") == "face_0")

# 4. Export filtered dataset
# high_confidence.export("/path/to/export", dataset_type=fo.types.COCODetectionDataset)