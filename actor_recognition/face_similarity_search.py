import os
import json
import numpy as np
import pandas as pd
import faiss

# Configuration
actor_embedding_path = "/data0/teja_codes/ImmersoAiResearch/ImageMatchingAlgo/actor_recognition/processed_images_actor/Shah_Rukh_Khan/Shah Rukh Khan_0_embeddings.json"
movie_frames_dir = "/data0/teja_codes/ImmersoAiResearch/ImageMatchingAlgo/actor_recognition/processed_images"
output_csv_path = "actor_similarity_results.csv"

# Load actor embedding
with open(actor_embedding_path, 'r') as f:
    actor_data = json.load(f)
actor_embedding = np.array(actor_data['face_0']['embedding'], dtype=np.float32).reshape(1, -1)

# Load movie frame embeddings
all_embeddings = []
meta_info = []

for root, _, files in os.walk(movie_frames_dir):
    for file in files:
        if file.endswith('.json'):
            file_path = os.path.join(root, file)
            try:
                with open(file_path, 'r') as f:
                    data = json.load(f)
                for face_id, face_data in data.items():
                    embedding = np.array(face_data['embedding'], dtype=np.float32)
                    all_embeddings.append(embedding)
                    meta_info.append({
                        "json_file": file_path,
                        "face_path": face_data["face_path"],
                        "original_image": face_data["original_image"]
                    })
            except Exception as e:
                print(f"Error reading {file_path}: {e}")

# Build FAISS index
all_embeddings_np = np.vstack(all_embeddings).astype(np.float32)
faiss.normalize_L2(all_embeddings_np)
faiss.normalize_L2(actor_embedding)

dim = all_embeddings_np.shape[1]
index = faiss.IndexFlatIP(dim)
index.add(all_embeddings_np)

# Perform similarity search
k = len(all_embeddings)
distances, indices = index.search(actor_embedding, k)


# distance bewtween 0 and 1
distances = (distances + 1) / 2

# Prepare results
results = []
for dist, idx in zip(distances[0], indices[0]):
    meta = meta_info[idx]
    results.append({
        "distance": dist,
        "face_path": meta["face_path"],
        "original_image": meta["original_image"],
        "json_file": meta["json_file"]
    })
    
# Save to CSV
df = pd.DataFrame(results)
df.to_csv(output_csv_path, index=False)
print(f"[✅] Saved similarity results to: {output_csv_path}")
