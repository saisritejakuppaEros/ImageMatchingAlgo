"""
Multi-Actor Face Search using FAISS
====================================

This script:
1. Loads all frame embeddings from pickle files
2. Stacks them into a numpy array and saves it
3. Builds a FAISS HNSW index for efficient similarity search (once)
4. Searches for MULTIPLE actor faces in the frame embeddings using cosine similarity
5. Generates individual CSV files for each actor
6. Creates a combined CSV with all actors' results

Features:
- Efficient batch processing: builds FAISS index once, searches multiple actors
- Individual results per actor for easy filtering
- Combined results CSV for comparative analysis
- Summary statistics showing match counts and similarity scores per actor

Usage:
    python actor_search.py
"""

import os
import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import faiss


def load_all_frame_embeddings(embeddings_root, csv_path):
    """
    Load all frame embeddings and create a mapping to face_id.
    
    Args:
        embeddings_root: Root directory containing frame embeddings
        csv_path: Path to face_detection.csv for metadata
        
    Returns:
        embeddings_array: numpy array of shape (N, embedding_dim)
        metadata_list: list of dicts with face_id, image_path, bbox, etc.
    """
    print("Loading frame embeddings...")
    
    # Read CSV to get all face_ids
    df = pd.read_csv(csv_path)
    print(f"Found {len(df)} face detections in CSV")
    
    embeddings_list = []
    metadata_list = []
    
    # Iterate through all pickle files
    pkl_files = list(Path(embeddings_root).rglob("*.pkl"))
    print(f"Found {len(pkl_files)} embedding files")
    
    for pkl_path in tqdm(pkl_files, desc="Loading embeddings"):
        try:
            with open(pkl_path, 'rb') as f:
                data = pickle.load(f)
            
            # Extract embedding and metadata
            embedding = np.array(data['embedding'])
            embeddings_list.append(embedding)
            
            metadata_list.append({
                'face_id': data['face_id'],
                'image_path': data['image_path'],
                'bbox': data['bbox'],
                'confidence': data['confidence'],
                'pkl_path': str(pkl_path)
            })
            
        except Exception as e:
            print(f"Error loading {pkl_path}: {e}")
            continue
    
    # Stack into numpy array
    embeddings_array = np.vstack(embeddings_list).astype('float32')
    print(f"Stacked embeddings shape: {embeddings_array.shape}")
    
    return embeddings_array, metadata_list


def normalize_embeddings(embeddings):
    """
    Normalize embeddings for cosine similarity.
    Cosine similarity = dot product of normalized vectors.
    """
    norms = np.linalg.norm(embeddings, axis=1, keepdims=True)
    return embeddings / (norms + 1e-8)


def save_embeddings_for_faiss(embeddings_array, output_dir):
    """
    Save embeddings in the format expected by autofaiss.
    
    Args:
        embeddings_array: numpy array of embeddings
        output_dir: directory to save the embeddings
    """
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "frame_embeddings.npy")
    
    print(f"Saving embeddings to {output_path}")
    np.save(output_path, embeddings_array)
    print(f"Saved embeddings with shape {embeddings_array.shape}")
    
    return output_path


def build_faiss_index_manual(embeddings_array, index_dir):
    """
    Build FAISS index manually (faster and more reliable than autofaiss).
    
    Args:
        embeddings_array: numpy array of embeddings (already normalized)
        index_dir: directory to save the index
        
    Returns:
        Path to the built index
    """
    print("\nBuilding FAISS index...")
    os.makedirs(index_dir, exist_ok=True)
    
    index_path = os.path.join(index_dir, "knn.index")
    
    # Get embedding dimension
    d = embeddings_array.shape[1]
    n = embeddings_array.shape[0]
    
    print(f"Building index for {n} vectors of dimension {d}")
    
    # Use IndexHNSWFlat for fast similarity search (good for < 1M vectors)
    # Using Inner Product (IP) metric for cosine similarity on normalized vectors
    index = faiss.IndexHNSWFlat(d, 32, faiss.METRIC_INNER_PRODUCT)
    index.hnsw.efConstruction = 40  # Higher = better quality, slower build
    
    print("Adding vectors to index...")
    index.add(embeddings_array)
    
    # Set search quality parameter
    index.hnsw.efSearch = 64  # Higher = better recall, slower search
    
    print(f"Saving index to {index_path}")
    faiss.write_index(index, index_path)
    
    print(f"Index built successfully with {index.ntotal} vectors")
    return index_path


def load_actor_embedding(actor_pkl_path):
    """
    Load actor embedding from pickle file.
    
    Args:
        actor_pkl_path: path to actor embedding pickle file
        
    Returns:
        embedding: numpy array
        metadata: dict with actor info
    """
    with open(actor_pkl_path, 'rb') as f:
        data = pickle.load(f)
    
    embedding = np.array(data['embedding']).astype('float32')
    return embedding, data


def get_actor_embeddings_from_directory(actor_dir):
    """
    Get all actor embedding paths from a directory.
    
    Args:
        actor_dir: directory containing actor embedding pickle files
        
    Returns:
        List of paths to actor embedding pickle files
    """
    actor_dir_path = Path(actor_dir)
    if not actor_dir_path.exists():
        print(f"Warning: Actor directory not found: {actor_dir}")
        return []
    
    actor_paths = sorted(list(actor_dir_path.glob("*.pkl")))
    print(f"Found {len(actor_paths)} actor embeddings in {actor_dir}")
    
    return [str(p) for p in actor_paths]


def search_actor_in_frames(index_path, actor_embedding, metadata_list, top_k=100):
    """
    Search for actor embedding in frame embeddings using FAISS index.
    
    Args:
        index_path: path to FAISS index
        actor_embedding: numpy array of actor embedding
        metadata_list: list of metadata dicts for frame embeddings
        top_k: number of top matches to return
        
    Returns:
        results: list of dicts with match info
    """
    print(f"\nSearching for actor in {len(metadata_list)} frame embeddings...")
    
    # Load FAISS index
    index = faiss.read_index(index_path)
    print(f"Loaded FAISS index with {index.ntotal} vectors")
    
    # Normalize actor embedding for cosine similarity
    actor_embedding_norm = normalize_embeddings(actor_embedding.reshape(1, -1))
    
    # Search
    distances, indices = index.search(actor_embedding_norm, top_k)
    
    # Prepare results
    results = []
    for i, (dist, idx) in enumerate(zip(distances[0], indices[0])):
        if idx < len(metadata_list):
            result = metadata_list[idx].copy()
            result['similarity_score'] = float(dist)
            result['rank'] = i + 1
            results.append(result)
    
    print(f"Found {len(results)} matches")
    return results


def create_results_csv(results, actor_info, output_csv):
    """
    Create CSV file with search results.
    
    Args:
        results: list of match results
        actor_info: dict with actor metadata
        output_csv: path to output CSV file
    
    Returns:
        DataFrame with results
    """
    print(f"\nCreating results CSV...")
    
    # Extract actor name from path
    actor_pkl_path = actor_info.get('pkl_path', 'unknown')
    actor_name = Path(actor_pkl_path).stem if actor_pkl_path != 'unknown' else 'unknown'
    
    # Prepare data for CSV
    csv_data = []
    for result in results:
        bbox = result['bbox']
        csv_data.append({
            'rank': result['rank'],
            'similarity_score': result['similarity_score'],
            'frame_image_path': result['image_path'],
            'face_x1': bbox['x1'],
            'face_y1': bbox['y1'],
            'face_x2': bbox['x2'],
            'face_y2': bbox['y2'],
            'face_confidence': result['confidence'],
            'face_id': result['face_id'],
            'actor_name': actor_name,
            'actor_embedding_path': actor_pkl_path
        })
    
    # Create DataFrame and save
    df = pd.DataFrame(csv_data)
    
    # Create output directory if it doesn't exist
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    
    df.to_csv(output_csv, index=False)
    print(f"Saved results to {output_csv}")
    print(f"Total matches: {len(df)}")
    print(f"\nTop 5 matches:")
    print(df.head()[['rank', 'similarity_score', 'frame_image_path', 'actor_name']])
    
    return df


def search_multiple_actors(index_path, actor_embedding_paths, metadata_list, output_dir, top_k=None):
    """
    Search for multiple actors in frames using the same FAISS index.
    
    Args:
        index_path: path to FAISS index
        actor_embedding_paths: list of paths to actor embedding pickle files
        metadata_list: list of metadata dicts for frame embeddings
        output_dir: directory to save individual results
        top_k: number of top matches to return per actor (None = all)
        
    Returns:
        combined_df: DataFrame with all actors' results combined
    """
    if top_k is None:
        top_k = len(metadata_list)
    
    all_results = []
    
    print("\n" + "="*80)
    print(f"SEARCHING FOR {len(actor_embedding_paths)} ACTORS")
    print("="*80)
    
    for i, actor_path in enumerate(actor_embedding_paths, 1):
        print(f"\n[{i}/{len(actor_embedding_paths)}] Processing: {Path(actor_path).stem}")
        print("-" * 60)
        
        # Load actor embedding
        actor_embedding, actor_data = load_actor_embedding(actor_path)
        actor_data['pkl_path'] = actor_path
        
        # Search for this actor
        results = search_actor_in_frames(index_path, actor_embedding, metadata_list, top_k=top_k)
        
        # Create individual CSV for this actor
        actor_name = Path(actor_path).stem
        output_csv = os.path.join(output_dir, f"actor_search_results_{actor_name}.csv")
        df = create_results_csv(results, actor_data, output_csv)
        
        # Add to combined results
        all_results.append(df)
    
    # Combine all results
    print("\n" + "="*80)
    print("COMBINING ALL RESULTS")
    print("="*80)
    combined_df = pd.concat(all_results, ignore_index=True)
    combined_output = os.path.join(output_dir, "actor_search_results_combined.csv")
    combined_df.to_csv(combined_output, index=False)
    print(f"Combined results saved to: {combined_output}")
    print(f"Total rows: {len(combined_df)}")
    
    # Summary statistics
    print("\n" + "="*80)
    print("SUMMARY BY ACTOR")
    print("="*80)
    summary = combined_df.groupby('actor_name').agg({
        'similarity_score': ['count', 'mean', 'max', 'min']
    }).round(4)
    print(summary)
    
    return combined_df


def main():
    """Main function to run actor search for multiple actors."""
    
    # ========== CONFIGURATION ==========
    
    # Frame embeddings configuration
    embeddings_root = 'output/devdas_embeddings'
    csv_path = 'output/devadas_faces.csv'
    
    # Actor embeddings to search for
    # Option 1: Manually specify actor embedding paths
    actor_embedding_paths = [
        '/data0/teja_works/dataset_captioning/actor_embeddings/aish_face_0.pkl',
        '/data0/teja_works/dataset_captioning/actor_embeddings/madhuri_face_1.pkl',
        '/data0/teja_works/dataset_captioning/actor_embeddings/srk_face_2.pkl'
    ]
    
    # Option 2: Or load all actors from a directory (uncomment to use)
    # actor_dir = '/data0/teja_works/dataset_captioning/actor_embeddings'
    # actor_embedding_paths = get_actor_embeddings_from_directory(actor_dir)
    
    # Output paths
    stacked_embeddings_dir = 'output/faiss/stacked_embeddings'
    index_dir = 'output/faiss/faiss_index'
    output_dir = 'output/faiss/results'
    
    # Search parameters
    top_k = None  # None = return all matches, or specify a number (e.g., 1000)
    
    # ===================================
    
    # Validate actors
    if not actor_embedding_paths:
        print("ERROR: No actor embedding paths provided!")
        print("Please specify actor_embedding_paths or use get_actor_embeddings_from_directory()")
        return
    
    print("="*80)
    print("MULTI-ACTOR FACE SEARCH USING FAISS")
    print("="*80)
    print(f"Frame embeddings root: {embeddings_root}")
    print(f"Face detection CSV: {csv_path}")
    print(f"Number of actors: {len(actor_embedding_paths)}")
    print(f"Actors:")
    for actor_path in actor_embedding_paths:
        print(f"  - {Path(actor_path).stem}")
    print(f"Output directory: {output_dir}")
    print("="*80)
    
    # Step 1: Load all frame embeddings (once for all actors)
    print("\n[STEP 1] Loading frame embeddings...")
    embeddings_array, metadata_list = load_all_frame_embeddings(embeddings_root, csv_path)
    
    # Step 2: Normalize embeddings for cosine similarity
    print("\n[STEP 2] Normalizing embeddings for cosine similarity...")
    embeddings_array_norm = normalize_embeddings(embeddings_array)
    
    # Step 3: Save embeddings for backup (optional)
    print("\n[STEP 3] Saving stacked embeddings...")
    embeddings_npy_path = save_embeddings_for_faiss(embeddings_array_norm, stacked_embeddings_dir)
    
    # Step 4: Build FAISS index (once for all actors)
    print("\n[STEP 4] Building FAISS index...")
    index_path = build_faiss_index_manual(embeddings_array_norm, index_dir)
    
    # Step 5: Search for all actors
    print("\n[STEP 5] Searching for actors...")
    combined_df = search_multiple_actors(
        index_path, 
        actor_embedding_paths, 
        metadata_list, 
        output_dir,
        top_k=top_k
    )
    
    print("\n" + "="*80)
    print("ALL SEARCHES COMPLETE!")
    print("="*80)
    print(f"\nResults saved to: {output_dir}")
    print(f"  - Individual CSVs: actor_search_results_<actor_name>.csv")
    print(f"  - Combined CSV: actor_search_results_combined.csv")
    print(f"FAISS index saved to: {index_path}")
    print(f"Stacked embeddings saved to: {embeddings_npy_path}")
    print("="*80)


if __name__ == "__main__":
    main()
