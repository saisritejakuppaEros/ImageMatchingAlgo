import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from transformers import CLIPModel, CLIPProcessor
from torch.multiprocessing import Process, set_start_method

def get_image_paths(directory):
    """Get all image paths recursively in a directory"""
    image_paths = []
    image_extensions = {'.jpg', '.jpeg', '.png'}
    
    for root, _, files in os.walk(directory):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(Path(root) / file)
    
    return image_paths

def process_images(gpu_id, region_paths):
    # Set the GPU device
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    
    # Load model and processor
    print(f"\nInitializing CLIP model on GPU {gpu_id}...")
    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    
    # Create base output directory
    output_base_dir = Path('place_clip_embeddings')
    output_base_dir.mkdir(parents=True, exist_ok=True)
    
    # Process each region and its states
    for region_path in tqdm(region_paths, desc=f'GPU {gpu_id} - Processing regions', position=gpu_id*2):
        region_name = region_path.name
        
        # Create output directory structure
        region_output_dir = output_base_dir / region_name
        region_output_dir.mkdir(parents=True, exist_ok=True)
        
        # Get all states in the region
        state_dirs = [d for d in region_path.iterdir() if d.is_dir()]
        
        # Process each state in the region
        for state_dir in tqdm(state_dirs, desc=f'GPU {gpu_id} - States in {region_name}', position=gpu_id*2+1, leave=False):
            if state_dir.is_dir():
                state_name = state_dir.name
                state_output_dir = region_output_dir / state_name
                state_output_dir.mkdir(exist_ok=True)
                
                # Get all images recursively in the state directory
                image_paths = get_image_paths(state_dir)
                
                total_images = len(image_paths)
                if total_images == 0:
                    print(f"No images found in {region_name}/{state_name}")
                    continue
                
                print(f"\nProcessing {total_images} images in {region_name}/{state_name}")
                
                # Process each image
                for img_path in tqdm(image_paths, desc=f'Images in {state_name}', leave=False):
                    try:
                        # Create subdirectories in output to match input structure
                        relative_path = img_path.relative_to(state_dir)
                        output_path = state_output_dir / relative_path.parent / f"{img_path.stem}.npy"
                        output_path.parent.mkdir(parents=True, exist_ok=True)
                        
                        # Skip if output already exists
                        if output_path.exists():
                            continue
                            
                        # Load and process image
                        img = Image.open(img_path).convert('RGB')
                        inputs = processor(images=img, return_tensors='pt', padding=True)
                        inputs = {k: v.to(device) for k, v in inputs.items()}
                        
                        # Generate embedding
                        with torch.no_grad():
                            vision_outputs = model.vision_model(**inputs)
                            image_embeds = vision_outputs[1]
                            image_embeds = model.visual_projection(image_embeds)
                            image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
                        
                        # Save embedding
                        np.save(output_path, image_embeds.cpu().numpy())
                        
                    except Exception as e:
                        print(f"\nError processing {img_path}: {str(e)}")

def main():
    # Enable parallel processing
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    
    # Set base directory for images
    base_dir = Path("/data0/indic_data/images/places")
    if not base_dir.exists():
        raise FileNotFoundError(f"Base directory not found: {base_dir}")
    
    # Get all region directories
    region_paths = [p for p in base_dir.iterdir() if p.is_dir() and not p.name.startswith('.')]
    
    if not region_paths:
        raise ValueError(f"No region directories found in {base_dir}")
    
    print(f"Found {len(region_paths)} regions to process")
    print("Regions found:", [p.name for p in region_paths])
    
    # Split regions between GPUs
    n_regions = len(region_paths)
    split_idx = n_regions // 2
    gpu_1_regions = region_paths[:split_idx]
    gpu_2_regions = region_paths[split_idx:]
    
    print(f"\nDistributing regions across GPUs:")
    print(f"GPU 0: {[r.name for r in gpu_1_regions]}")
    print(f"GPU 1: {[r.name for r in gpu_2_regions]}")
    
    # Create processes for each GPU
    p1 = Process(target=process_images, args=(0, gpu_1_regions))
    p2 = Process(target=process_images, args=(1, gpu_2_regions))
    
    print("\nStarting parallel processing...")
    
    # Start processes
    p1.start()
    p2.start()
    
    # Wait for completion
    p1.join()
    p2.join()
    
    print("\nAll regions processed successfully!")

if __name__ == '__main__':
    main()