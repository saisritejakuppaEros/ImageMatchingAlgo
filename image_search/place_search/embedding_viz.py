places_dir = '/data0/teja_codes/ImmersoAiResearch/ImageMatchingAlgo/image_search/place_search'
frames_dir = '/data0/sriprabha/codes/SimpleTuner/datasets/eros_10lakh/train'


import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
import umap
from tqdm import tqdm
import concurrent.futures

def collect_embeddings(root_dir, label_tag):
    paths, labels = [], []
    for dirpath, _, filenames in os.walk(root_dir):
        for fname in filenames:
            if fname.endswith('.npy'):
                paths.append(os.path.join(dirpath, fname))
                labels.append(label_tag)
    return paths, labels

# Step 1: Walk and collect .npy paths with labels
place_paths, place_labels = collect_embeddings(places_dir, "place")
frame_paths, frame_labels = collect_embeddings(frames_dir, "frame")

all_paths = place_paths + frame_paths
all_labels = place_labels + frame_labels

# Step 2: Load all embeddings in parallel
def load_embedding(path_label):
    path, label = path_label
    try:
        emb = np.load(path)
        if emb.ndim == 1:
            return [emb], [label]
        elif emb.ndim == 2:
            return list(emb), [label] * len(emb)
    except Exception as e:
        print(f"Error loading {path}: {e}")
        return [], []

all_embeddings = []
final_labels = []

with concurrent.futures.ThreadPoolExecutor() as executor:
    results = list(tqdm(executor.map(load_embedding, zip(all_paths, all_labels)), total=len(all_paths)))

for embs, labels in results:
    all_embeddings.extend(embs)
    final_labels.extend(labels)

all_embeddings = np.array(all_embeddings)
print(f"Total embeddings: {all_embeddings.shape[0]}, Dim: {all_embeddings.shape[1]}")

import time
start_time = time.time()


# Step 3: PCA to 50D
pca = PCA(n_components=50)
pca_result = pca.fit_transform(all_embeddings)

# Step 4: UMAP to 2D
umap_model = umap.UMAP(n_components=2, random_state=42)
umap_result = umap_model.fit_transform(pca_result)

end_time = time.time()
print(f"Time taken: {end_time - start_time} seconds")






