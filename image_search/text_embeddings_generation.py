"""
embeddings_generation.py
-----------------------
This script extracts the 'character_subject' field from all JSON files in the captions directory, generates sentence embeddings using SentenceTransformer, and saves the embeddings as a .npy file.

Usage:
    python embeddings_generation.py

Configuration:
    - Configuration is specified in config.toml
    - GPU IDs, captions path, and output file are configurable
    - Embeddings are saved to the specified output file
"""
import os
import glob
import json
import toml
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import Dataset, DataLoader
import torch.multiprocessing as mp

class TextDataset(Dataset):
    def __init__(self, texts):
        self.texts = texts
    
    def __len__(self):
        return len(self.texts)
    
    def __getitem__(self, idx):
        return self.texts[idx]

def load_config(config_path='config.toml'):
    with open(config_path, 'r') as f:
        return toml.load(f)

def process_batch_on_gpu(gpu_id, texts, model_name, start_idx, end_idx, batch_size, shared_tensor):
    """Process a batch of texts on a specific GPU using batched processing"""
    try:
        device = f'cuda:{gpu_id}'
        model = SentenceTransformer(model_name, device=device)
        
        # Create dataset and dataloader for batched processing
        dataset = TextDataset(texts[start_idx:end_idx])
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        current_idx = 0
        with tqdm(total=end_idx-start_idx, 
                  desc=f'GPU {gpu_id} inference', 
                  position=gpu_id,
                  leave=True) as pbar:
            for batch in dataloader:
                # Process batch
                emb = model.encode(batch, batch_size=len(batch))
                batch_size_actual = len(emb)
                
                # Store in shared tensor
                shared_tensor[start_idx + current_idx:start_idx + current_idx + batch_size_actual].copy_(
                    torch.tensor(emb, device='cpu')
                )
                current_idx += batch_size_actual
                
                pbar.update(batch_size_actual)
                pbar.set_postfix({'batch_size': batch_size_actual})
        
    except Exception as e:
        print(f"Error in GPU {gpu_id} process: {str(e)}")
        raise e

def main():
    config = load_config()
    gpu_ids = config.get('gpu_ids', [0, 1])
    captions_path = config.get('captions_path', 'captions')
    model_name = config.get('model_name', 'sentence-transformers/all-MiniLM-L6-v2')
    output_file = config.get('output_file', 'character_embeddings.npy')
    batch_size = config.get('batch_size', 16)
    
    # Create text_embeddings directory if it doesn't exist
    embeddings_dir = "text_embeddings"
    os.makedirs(embeddings_dir, exist_ok=True)
    
    # Check available GPUs
    if not torch.cuda.is_available():
        print("CUDA not available, using CPU")
        gpu_ids = []
    
    print(f"Available GPUs: {torch.cuda.device_count()}")
    print(f"Using GPU IDs: {gpu_ids}")
    print(f"Batch size: {batch_size}")

    # Step 1: Find all JSON files in the captions directory
    print("\nStep 1: Loading character subjects")
    json_files = glob.glob(f'{captions_path}/*.json')
    
    # sort the json files by the file name
    json_files.sort()
    character_subjects = []
    file_names = []
    for json_file in tqdm(json_files, desc='Loading character_subjects'):
        with open(json_file, 'r') as f:
            data = json.load(f)
            character_subject = data.get('character_subject', None)
            if character_subject is not None:
                character_subjects.append(character_subject)
                file_names.append(json_file)

    print(f"\nFound {len(character_subjects)} character subjects to process")

    # Step 2: Process embeddings
    print("\nStep 2: Processing embeddings")
    if len(gpu_ids) > 1:
        # Multi-GPU processing
        num_texts = len(character_subjects)
        num_gpus = len(gpu_ids)
        texts_per_gpu = num_texts // num_gpus
        
        # Get embedding dimension by running a single example
        temp_model = SentenceTransformer(model_name)
        temp_emb = temp_model.encode(["test"])
        embedding_dim = temp_emb.shape[1]
        del temp_model
        
        # Create batches for each GPU
        batches = []
        for i, gpu_id in enumerate(gpu_ids):
            start_idx = i * texts_per_gpu
            if i == num_gpus - 1:  # Last GPU gets remaining texts
                end_idx = num_texts
            else:
                end_idx = (i + 1) * texts_per_gpu
            batches.append((gpu_id, start_idx, end_idx))
        
        print(f"\nDistributing {num_texts} texts across {num_gpus} GPUs")
        for gpu_id, start_idx, end_idx in batches:
            num_batches = (end_idx - start_idx + batch_size - 1) // batch_size
            print(f"GPU {gpu_id}: texts {start_idx} to {end_idx-1} ({end_idx-start_idx} texts, {num_batches} batches)")

        # Create shared tensor for all embeddings
        shared_tensor = torch.zeros((num_texts, embedding_dim), dtype=torch.float32).share_memory_()
        
        print("\nStarting parallel inference on multiple GPUs:")
        # Create processes for each GPU
        mp.set_start_method('spawn', force=True)
        
        processes = []
        for gpu_id, start_idx, end_idx in batches:
            p = mp.Process(
                target=process_batch_on_gpu, 
                args=(gpu_id, character_subjects, model_name, start_idx, end_idx, 
                      batch_size, shared_tensor)
            )
            p.start()
            processes.append(p)
        
        # Wait for all processes to complete
        for p in processes:
            p.join()
        
        # Convert shared tensor to numpy array
        embeddings = shared_tensor.numpy()
        
    else:
        # Single GPU or CPU processing
        device = f'cuda:{gpu_ids[0]}' if gpu_ids else 'cpu'
        print(f"\nProcessing on single {device}")
        
        model = SentenceTransformer(model_name, device=device)
        
        # Create dataset and dataloader for batched processing
        dataset = TextDataset(character_subjects)
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
        embeddings = []
        with tqdm(total=len(character_subjects), 
                 desc=f'{device} inference', 
                 position=0,
                 leave=True) as pbar:
            for batch in dataloader:
                emb = model.encode(batch, batch_size=len(batch))
                embeddings.extend(emb)
                pbar.update(len(batch))
                pbar.set_postfix({'batch_size': len(batch)})
        
        embeddings = np.array(embeddings)

    # Step 3: Save embeddings
    print("\nStep 3: Saving embeddings")
    # Save the complete embeddings array
    np.save(output_file, embeddings)
    print(f"✓ Saved complete embeddings to {output_file}. Shape: {embeddings.shape}")
    
    # Save individual embeddings with corresponding filenames
    print("\nSaving individual embeddings...")
    for idx, json_file in enumerate(tqdm(file_names, desc='Saving individual embeddings')):
        # Get the base filename without extension and path
        base_name = os.path.splitext(os.path.basename(json_file))[0]
        # Create the output path
        output_path = os.path.join(embeddings_dir, f"{base_name}.npy")
        # Save the individual embedding
        np.save(output_path, embeddings[idx])
    
    print(f"✓ Saved {len(file_names)} individual embeddings in {embeddings_dir}/")

if __name__ == "__main__":
    main() 