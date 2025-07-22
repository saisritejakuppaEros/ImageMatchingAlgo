search_embedding = '/data0/teja_codes/ImmersoAiResearch/ImageMatchingAlgo/image_search/place_search/place_clip_embeddings/South_India/Andhra_Pradesh/Rajahmundry/image_6.npy'
search_image = '/data0/indic_data/images/places/South_India/Andhra_Pradesh/Rajahmundry/image_6.jpg'

all_embeddings = []
import os
import numpy as np
emd_path = '/data0/teja_dataset_embeddings/clip_embeddings'
embedding_paths = [os.path.join(emd_path, f) for f in os.listdir(emd_path) if f.endswith('.npy')]


#shuffle the embedding paths
import random
random.shuffle(embedding_paths)

from tqdm import tqdm

#load the embeddings
for emb in tqdm(embedding_paths[:500]):
    embedding = np.load(emb)
    all_embeddings.append(embedding)

all_embeddings = np.array(all_embeddings)
all_embeddings = all_embeddings.reshape(500, 768)
print("total embeddings loaded", all_embeddings.shape)

# check using cosine similarity
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

query_embedding = np.load('/data0/teja_codes/ImmersoAiResearch/ImageMatchingAlgo/image_search/place_search/place_clip_embeddings/South_India/Andhra_Pradesh/Rajahmundry/image_6.npy')

similarities = cosine_similarity(query_embedding, all_embeddings)


# get me the index of the embedding with similarty > 0.8
indices = np.where(similarities > 0.85)[1]


# the images are located at /data0/sriprabha/codes/SimpleTuner/datasets/eros_10lakh/train
image_path = '/data0/sriprabha/codes/SimpleTuner/datasets/eros_10lakh/train'

# the images with jpg image from the embedding_paths[:500]
image_paths = [os.path.join(image_path, os.path.basename(f).replace('.npy', '.png')) for f in embedding_paths[:500]]

# !pip install fiftyone
# get the imges in the indices
filtered_indices = np.where(similarities > 0.52)[1]
img_filtered_paths = [image_paths[i] for i in filtered_indices]

import fiftyone as fo

# Create a dataset from a list of images
dataset = fo.Dataset.from_images(
    img_filtered_paths,
)

session = fo.launch_app(dataset)

while True:
    pass