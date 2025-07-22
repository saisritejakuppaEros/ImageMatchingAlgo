import os
import torch
import numpy as np
from PIL import Image
from tqdm import tqdm
from pathlib import Path
from transformers import CLIPModel, CLIPProcessor
from torch.multiprocessing import Process, set_start_method

def process_images(gpu_id, image_paths):
    # Set the GPU device
    torch.cuda.set_device(gpu_id)
    device = torch.device(f'cuda:{gpu_id}')
    
    # Load model and processor
    model = CLIPModel.from_pretrained('openai/clip-vit-large-patch14').to(device)
    processor = CLIPProcessor.from_pretrained('openai/clip-vit-large-patch14')
    
    # Create output directory if it doesn't exist
    output_dir = Path('clip_embeddings')
    output_dir.mkdir(exist_ok=True)
    
    # Process each image
    for img_path in tqdm(image_paths, desc=f'Processing on GPU {gpu_id}'):
        try:
            # Skip if output already exists
            output_path = output_dir / f"{img_path.stem}.npy"
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
            print(f"Error processing {img_path}: {str(e)}")

def main():
    # Enable parallel processing
    try:
        set_start_method('spawn')
    except RuntimeError:
        pass
    
    # Get all image paths
    img_dir = Path("/data0/sriprabha/codes/SimpleTuner/datasets/eros_10lakh/trainnew")
    image_paths = list(img_dir.glob('*.jpg')) + list(img_dir.glob('*.jpeg')) + list(img_dir.glob('*.png'))
    
    # Split images between GPUs
    n_images = len(image_paths)
    split_idx = n_images // 2
    gpu_3_images = image_paths[:split_idx]
    gpu_4_images = image_paths[split_idx:]
    
    # Create processes for each GPU
    p1 = Process(target=process_images, args=(1, gpu_3_images))
    p2 = Process(target=process_images, args=(2, gpu_4_images))
    
    # Start processes
    p1.start()
    p2.start()
    
    # Wait for completion
    p1.join()
    p2.join()

if __name__ == '__main__':
    main()




