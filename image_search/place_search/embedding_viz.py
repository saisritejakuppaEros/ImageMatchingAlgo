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
        
        # Files to skip (common non-embedding files)
        skip_files = {'final_labels.npy', 'labels.npy', 'metadata.npy', 'index.npy'}
        
        print(f"Scanning directory: {root_dir}")
        for dirpath, _, filenames in tqdm(os.walk(root_dir), desc=f"Scanning {label_tag} dirs"):
            npy_files = [f for f in filenames if f.endswith('.npy') and f not in skip_files]
            for fname in npy_files:
                paths.append(os.path.join(dirpath, fname))
                labels.append(label_tag)
        
        return paths, labels

    def load_single_embedding(self, path_label: Tuple[str, str]) -> Tuple[np.ndarray, List[str]]:
        """Load a single embedding file with error handling"""
        path, label = path_label
        try:
            emb = np.load(path)
            # Expected CLIP embedding dimension
            EXPECTED_DIM = 768
            
            if emb.ndim == 1:
                if emb.shape[0] != EXPECTED_DIM:
                    logger.warning(f"Skipping {path}: Wrong dimension {emb.shape[0]} (expected {EXPECTED_DIM})")
                    return np.array([]), []
                # Ensure 1D embeddings are reshaped to 2D
                return emb.reshape(1, -1), [label]
            elif emb.ndim == 2:
                if emb.shape[1] != EXPECTED_DIM:
                    logger.warning(f"Skipping {path}: Wrong dimension {emb.shape[1]} (expected {EXPECTED_DIM})")
                    return np.array([]), []
                # Return the 2D embedding directly
                return emb, [label] * emb.shape[0]
            else:
                logger.warning(f"Skipping {path}: Unexpected shape {emb.shape}")
                return np.array([]), []
        except Exception as e:
            logger.error(f"Error loading {path}: {e}")
            return np.array([]), []

    def load_embeddings_batch(self, path_labels: List[Tuple[str, str]]) -> Tuple[List[np.ndarray], List[str], int]:
        """Load a batch of embeddings in parallel"""
        batch_embeddings = []
        batch_labels = []
        files_processed = 0
        files_skipped = 0
        
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
                    files_processed += 1
                    if embs.size > 0:  # Check if we got valid embeddings
                        batch_embeddings.append(embs)
                        batch_labels.extend(labels)
                        # Debug: print info about multi-embedding files
                        if embs.shape[0] > 1:
                            path = futures[future][0]
                            logger.info(f"File {path} contains {embs.shape[0]} embeddings")
                    else:
                        files_skipped += 1
                except Exception as e:
                    path, label = futures[future]
                    logger.error(f"Failed to process {path}: {e}")
                    files_processed += 1
                    files_skipped += 1
        
        return batch_embeddings, batch_labels, files_skipped

    def load_all_embeddings(self, all_paths: List[str], all_labels: List[str]) -> Tuple[np.ndarray, List[str]]:
        """Load all embeddings in batches with memory management"""
        total_files = len(all_paths)
        all_embeddings = []
        final_labels = []
        total_skipped = 0
        files_with_multiple_embeddings = 0
        
        print(f"Loading {total_files:,} embedding files in batches of {self.batch_size:,}")
        print(f"Using {self.max_workers} workers")
        
        # Process in batches to manage memory
        for i in tqdm(range(0, total_files, self.batch_size), desc="Processing batches"):
            batch_end = min(i + self.batch_size, total_files)
            batch_paths = all_paths[i:batch_end]
            batch_labels = all_labels[i:batch_end]
            
            path_labels = list(zip(batch_paths, batch_labels))
            
            # Load batch
            batch_embeddings, batch_final_labels, batch_skipped = self.load_embeddings_batch(path_labels)
            
            # Count files with multiple embeddings
            for emb in batch_embeddings:
                if emb.shape[0] > 1:
                    files_with_multiple_embeddings += 1
            
            total_skipped += batch_skipped
            
            # Extend results
            all_embeddings.extend(batch_embeddings)
            final_labels.extend(batch_final_labels)
            
            # Memory management
            if i > 0 and i % (self.batch_size * 10) == 0:  # Every 10 batches
                gc.collect()
                current_count = sum(emb.shape[0] for emb in all_embeddings if emb.size > 0)
                print(f"Processed {current_count:,} embeddings so far...")
        
        # Convert list of embeddings to a single numpy array
        print("Converting to numpy array...")
        if not all_embeddings:
            raise ValueError("No valid embeddings were loaded!")
        
        # Filter out empty arrays
        valid_embeddings = [emb for emb in all_embeddings if emb.size > 0]
        
        if not valid_embeddings:
            raise ValueError("No valid embeddings were loaded!")
        
        total_embedding_count = sum(emb.shape[0] for emb in valid_embeddings)
        valid_files = len(valid_embeddings)
        
        print(f"Files processed: {total_files:,}")
        print(f"Valid files loaded: {valid_files:,}")
        print(f"Files skipped: {total_skipped:,}")
        print(f"Files with multiple embeddings: {files_with_multiple_embeddings:,}")
        print(f"Total individual embeddings: {total_embedding_count:,}")
        print(f"Average embeddings per valid file: {total_embedding_count/valid_files:.2f}")
        
        # Concatenate all embeddings
        final_embeddings = np.concatenate(valid_embeddings, axis=0)
        
        # Convert labels to numpy array for proper indexing
        final_labels_array = np.array(final_labels)
        
        print(f"Before deduplication: {final_embeddings.shape[0]:,} embeddings")
        
        # Remove duplicate embeddings
        print("Removing duplicate embeddings...")
        unique_embeddings, unique_indices = np.unique(final_embeddings, axis=0, return_index=True)
        
        # Keep labels corresponding to unique embeddings
        unique_labels = final_labels_array[unique_indices].tolist()
        
        duplicates_removed = final_embeddings.shape[0] - unique_embeddings.shape[0]
        print(f"Removed {duplicates_removed:,} duplicate embeddings")
        print(f"After deduplication: {unique_embeddings.shape[0]:,} unique embeddings")
        
        print(f"Final embeddings shape: {unique_embeddings.shape}")
        
        return unique_embeddings, unique_labels

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
    
    # np.save('all_embeddings.npy', all_embeddings)
    # np.save('final_labels.npy', final_labels)
    
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
    print(f"UMAP result: {umap_result.shape}")

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
    os.environ['OMP_NUM_THREADS'] = str(int(os.cpu_count() * 0.25))
    os.environ['MKL_NUM_THREADS'] = str(int(os.cpu_count() * 0.25))
    
    umap_result, final_labels = main()