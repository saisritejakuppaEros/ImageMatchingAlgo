import numpy as np
import os
import re
from pathlib import Path
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor
import psutil
from tqdm.auto import tqdm
import time
import faiss
import plotly.graph_objects as go
from sklearn.decomposition import PCA
import umap
import pandas as pd

# Set thread limitations for numerical libraries
os.environ['OPENBLAS_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'
os.environ['NUMEXPR_NUM_THREADS'] = '1'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['BLAS_NUM_THREADS'] = '1'

# Set paths
BASE_EMBEDDINGS_DIR = "teja_dataset_embeddings"
EMBEDDING_DIRS = {
    'text': {
        # 'background_description': os.path.join(BASE_EMBEDDINGS_DIR, "text_embeddings/background_description"),
        # 'character_subject': os.path.join(BASE_EMBEDDINGS_DIR, "text_embeddings/character_subject"),
        # 'culture': os.path.join(BASE_EMBEDDINGS_DIR, "text_embeddings/culture"),
        'frame_description': os.path.join(BASE_EMBEDDINGS_DIR, "text_embeddings/frame_description")
    },
    'clip': {
        # 'image': os.path.join(BASE_EMBEDDINGS_DIR, "clip_embeddings")
    }
}

OUTPUT_DIR = os.path.join(BASE_EMBEDDINGS_DIR, "clustering_analysis")
os.makedirs(OUTPUT_DIR, exist_ok=True)

import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

# Helper functions
def extract_movie_and_frame(filename):
    """Extract movie name and frame number from filename."""
    # Extract frame number
    frame_match = re.search(r'frame_(\d+)', filename)
    frame_num = int(frame_match.group(1)) if frame_match else None
    
    # Extract movie name (everything before frame_xxx)
    movie_name = filename.split('frame_')[0].rstrip('_')
    
    return movie_name, frame_num

def load_single_embedding(file_path):
    """Load a single embedding file."""
    try:
        filename = os.path.basename(file_path)
        movie_name, frame_num = extract_movie_and_frame(filename)
        if frame_num is not None:
            embedding = np.load(file_path)
            return embedding, frame_num, movie_name, filename
    except Exception as e:
        print(f"Error loading {file_path}: {e}")
    return None

def chunk_list(lst, n):
    """Split list into n chunks."""
    chunk_size = max(1, len(lst) // n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]

def process_embeddings_directory(embedding_dir, embedding_type, subtype):
    """Process embeddings from a directory and save clustering results."""
    print(f"\nProcessing {embedding_type}/{subtype} embeddings from: {embedding_dir}")
    
    # Load embeddings
    N_CORES = max(1, int(psutil.cpu_count(logical=False) * 0.75))
    start_time = time.time()

    files = sorted([os.path.join(embedding_dir, f) for f in os.listdir(embedding_dir) 
                   if f.endswith('.npy')])

    if not files:
        print(f"No .npy files found in {embedding_dir}")
        return

    # Split files into chunks for parallel processing
    file_chunks = chunk_list(files, N_CORES)
    print(f"\nLoading {len(files)} embedding files using {N_CORES} cores...")

    results = []
    with ProcessPoolExecutor(max_workers=N_CORES) as executor:
        futures = []
        for chunk in file_chunks:
            futures.extend([executor.submit(load_single_embedding, f) for f in chunk])
        
        # Process results as they complete
        for future in tqdm(futures, total=len(files), desc="Loading embeddings", unit="files"):
            try:
                result = future.result()
                if result is not None:
                    results.append(result)
            except Exception as e:
                print(f"Error processing file: {e}")

    if not results:
        print(f"No embeddings could be loaded successfully from {embedding_dir}")
        return

    # Unzip the results
    embeddings, frame_numbers, movie_names, filenames = zip(*results)
    embeddings = np.vstack(embeddings)
    
    print(f"\nEmbeddings shape: {embeddings.shape}")
    
    # Initialize Faiss index and perform clustering
    print("Building Faiss index and performing clustering...")
    start_time = time.time()

    # Convert embeddings to float32
    embeddings = embeddings.astype(np.float32)

    # Create Faiss index
    d = embeddings.shape[1]  # dimension
    n_clusters = 100

    # Create and train k-means
    kmeans = faiss.Kmeans(d, n_clusters, niter=20, verbose=True, gpu=True)
    kmeans.train(embeddings)

    # Get cluster assignments
    _, clusters = kmeans.index.search(embeddings, 1)
    clusters = clusters.ravel()

    # Save results to CSV
    output_filename = f"clusters_{embedding_type}_{subtype}.csv"
    output_path = os.path.join(OUTPUT_DIR, output_filename)
    
    df = pd.DataFrame({
        'filename': filenames,
        'cluster': clusters
    })
    
    df.to_csv(output_path, index=False)
    print(f"\nClustering results saved to: {output_path}")
    
    elapsed = time.time() - start_time
    print(f"Clustering completed in {elapsed:.2f} seconds")

def main():
    """Process all embedding directories."""
    for embedding_type, subtypes in EMBEDDING_DIRS.items():
        for subtype, dir_path in subtypes.items():
            process_embeddings_directory(dir_path, embedding_type, subtype)

if __name__ == "__main__":
    main()
 
