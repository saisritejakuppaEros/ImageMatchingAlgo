import fiftyone as fo
import glob
import os
import random
import fiftyone as fo
import fiftyone.brain as fob
import fiftyone.zoo as foz
from fiftyone import ViewField as F


def get_image_paths(root_dir, limit=None):
    """Get all image file paths from a directory using os.walk"""
    image_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}
    image_paths = []
    
    for root, dirs, files in os.walk(root_dir):
        for file in files:
            if any(file.lower().endswith(ext) for ext in image_extensions):
                image_paths.append(os.path.join(root, file))
    
    if limit:
        return image_paths[:limit]
    return image_paths


# the images are located at /data0/sriprabha/codes/SimpleTuner/datasets/eros_10lakh/train
image_path = '/data0/sriprabha/codes/SimpleTuner/datasets/eros_10lakh/train'
img_paths = get_image_paths(image_path)



# shuffle the img_paths
random.shuffle(img_paths)
img_paths = img_paths[:100000]

# Create a dataset from a list of images
dataset = fo.Dataset.from_images(
    img_paths,
)


place1_path = '/data0/indic_data/images/places'
place1_paths = get_image_paths(place1_path)




dataset2 = fo.Dataset.from_images(
    place1_paths,
)


# Add a tag "d1" to all samples in the first dataset
for sample in dataset:
    sample.tags.append("d1")
    sample.save()

# Add a tag "d2" to all samples in the second dataset
for sample in dataset2:
    sample.tags.append("d2")
    sample.save()
    
    
import fiftyone as fo
final_dataset = fo.Dataset()
final_dataset.add_samples(dataset)
final_dataset.add_samples(dataset2)


res = fob.compute_visualization(
    final_dataset,
    model="clip-vit-base32-torch",
    embeddings="clip_embeddings",
    method="umap",
    brain_key="clip_vis",
    batch_size=64
)

# res2 = fob.compute_visualization(
#     final_dataset,
#     model="dinov2-vitb14-torch",
#     embeddings="dino_embeddings",
#     method="umap",
#     brain_key="dino_vis",
#     batch_size=64
# )
final_dataset.set_values("clip_umap", res.current_points)
# final_dataset.set_values("dino_umap", res2.current_points)

session = fo.launch_app(final_dataset)

while True:
    pass