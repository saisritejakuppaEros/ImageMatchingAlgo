import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
from tqdm import tqdm
from tqdm.contrib.concurrent import thread_map
import concurrent.futures
import time
from logzero import logger
import gc
from typing import List, Tuple, Optional
import threading

places_dir = '/data0/teja_codes/ImmersoAiResearch/ImageMatchingAlgo/image_search/place_search'
frames_dir = '/data0/teja_dataset_embeddings/clip_embeddings'

class EmbeddingLoader:
    def __init__(self, max_workers: Optional[int] = None, batch_size: int = 1000):
        self.max_workers = max_workers or min(32, (os.cpu_count() or 1) + 4)
        self.batch_size = batch_size
        self.lock = threading.Lock()
        
    def collect_embeddings(self, root_dir: str, label_tag: str) -> Tuple[List[str], List[str]]:
        """Collect all .npy file paths with their labels"""
        paths, labels = [], []
        
        print(f"Scanning directory: {root_dir}")
        for dirpath, _, filenames in tqdm(os.walk(root_dir), desc=f"Scanning {label_tag} dirs"):
            npy_files = [f for f in filenames if f.endswith('.npy')]
            for fname in npy_files:
                paths.append(os.path.join(dirpath, fname))
                labels.append(label_tag)
        
        return paths, labels

    def load_single_embedding(self, path_label: Tuple[str, str]) -> Tuple[List[np.ndarray], List[str]]:
        """Load a single embedding file with error handling"""
        path, label = path_label
        try:
            emb = np.load(path)
            if emb.ndim == 1:
                return [emb], [label]
            elif emb.ndim == 2:
                return list(emb), [label] * len(emb)
            else:
                logger.warning(f"Unexpected embedding shape {emb.shape} in {path}")
                return [], []
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return [], []

    def load_embeddings_batch(self, path_labels: List[Tuple[str, str]]) -> Tuple[List[np.ndarray], List[str]]:
        """Load a batch of embeddings in parallel"""
        batch_embeddings = []
        batch_labels = []
        
        # Use ThreadPoolExecutor for better control over memory and progress
        with concurrent.futures.ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all tasks
            futures = {executor.submit(self.load_single_embedding, pl): pl for pl in path_labels}
            
            # Process completed futures with progress bar
            for future in tqdm(concurrent.futures.as_completed(futures), 
                             total=len(futures), 
                             desc="Loading batch",
                             leave=False):
                try:
                    embs, labels = future.result()
                    batch_embeddings.extend(embs)
                    batch_labels.extend(labels)
                except Exception as e:
                    path, label = futures[future]
                    logger.error(f"Failed to process {path}: {e}")
        
        return batch_embeddings, batch_labels

    def load_all_embeddings(self, all_paths: List[str], all_labels: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Load all embeddings in batches with memory management"""
        total_files = len(all_paths)
        all_embeddings = []
        final_labels = []
        
        print(f"Loading {total_files:,} embedding files in batches of {self.batch_size:,}")
        print(f"Using {self.max_workers} workers")
        
        # Process in batches to manage memory
        for i in tqdm(range(0, total_files, self.batch_size), desc="Processing batches"):
            batch_end = min(i + self.batch_size, total_files)
            batch_paths = all_paths[i:batch_end]
            batch_labels = all_labels[i:batch_end]
            
            path_labels = list(zip(batch_paths, batch_labels))
            
            # Load batch
            batch_embeddings, batch_final_labels = self.load_embeddings_batch(path_labels)
            
            # Extend results
            all_embeddings.extend(batch_embeddings)
            final_labels.extend(batch_final_labels)
            
            # Memory management
            if i > 0 and i % (self.batch_size * 10) == 0:  # Every 10 batches
                gc.collect()
                current_count = len(all_embeddings)
                print(f"Processed {current_count:,} embeddings so far...")
        
        # Convert to numpy array
        print("Converting to numpy array...")
        all_embeddings = np.array(all_embeddings)
        
        return all_embeddings, final_labels

def main():
    # Initialize loader
    loader = EmbeddingLoader(max_workers=32, batch_size=2000)  # Adjust based on your system
    
    # Step 1: Collect all paths
    print("=" * 50)
    print("STEP 1: Collecting file paths")
    print("=" * 50)
    
    place_paths, place_labels = loader.collect_embeddings(places_dir, "place")
    frame_paths, frame_labels = loader.collect_embeddings(frames_dir, "frame")
    frame_paths = frame_paths[:50000]
    frame_labels = frame_labels[:50000]
    
    print(f"Found {len(place_paths):,} place files")
    print(f"Found {len(frame_paths):,} frame files")
    print(f"Total files: {len(place_paths) + len(frame_paths):,}")
    
    all_paths = place_paths + frame_paths
    all_labels = place_labels + frame_labels
    
    # Step 2: Load all embeddings
    print("\n" + "=" * 50)
    print("STEP 2: Loading embeddings")
    print("=" * 50)
    
    start_time = time.time()
    all_embeddings, final_labels = loader.load_all_embeddings(all_paths, all_labels)
    
    np.save('all_embeddings.npy', all_embeddings)
    np.save('final_labels.npy', final_labels)
    
    loading_time = time.time() - start_time
    
    print(f"\nLoading completed!")
    print(f"Total embeddings: {all_embeddings.shape[0]:,}")
    print(f"Embedding dimension: {all_embeddings.shape[1]}")
    print(f"Loading time: {loading_time:.2f} seconds")
    print(f"Loading rate: {len(all_embeddings)/loading_time:.1f} embeddings/second")
    
    # Step 3: PCA
    print("\n" + "=" * 50)
    print("STEP 3: Running PCA")
    print("=" * 50)
    
    pca_start = time.time()
    pca = PCA(n_components=50)
    pca_result = pca.fit_transform(all_embeddings)
    pca_time = time.time() - pca_start
    
    logger.info(f"PCA completed in {pca_time:.2f} seconds")
    print(f"PCA result shape: {pca_result.shape}")
    
    # Clear memory
    del all_embeddings
    gc.collect()
    
    # Step 4: UMAP
    print("\n" + "=" * 50)
    print("STEP 4: Running UMAP")
    print("=" * 50)
    
    umap_start = time.time()
    umap_model = umap.UMAP(
        n_components=2, 
        random_state=42,
        n_jobs=-1,  # Use all available cores
        verbose=True  # Show UMAP progress
    )
    umap_result = umap_model.fit_transform(pca_result)
    umap_time = time.time() - umap_start
    
    logger.info(f"UMAP completed in {umap_time:.2f} seconds")
    print(f"UMAP result shape: {umap_result.shape}")
    
    # Total time
    total_time = time.time() - start_time
    print(f"\n" + "=" * 50)
    print("SUMMARY")
    print("=" * 50)
    print(f"Total processing time: {total_time:.2f} seconds ({total_time/60:.1f} minutes)")
    print(f"Loading time: {loading_time:.2f} seconds ({loading_time/total_time*100:.1f}%)")
    print(f"PCA time: {pca_time:.2f} seconds ({pca_time/total_time*100:.1f}%)")
    print(f"UMAP time: {umap_time:.2f} seconds ({umap_time/total_time*100:.1f}%)")
    
    return umap_result, final_labels

if __name__ == "__main__":
    # Set numpy to use multiple threads
    os.environ['OMP_NUM_THREADS'] = str(int(os.cpu_count()) * 0.25)
    os.environ['MKL_NUM_THREADS'] = str(int(os.cpu_count()) * 0.25)
    
    umap_result, final_labels = main()