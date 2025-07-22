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
img_paths = img_paths[:10]

# Create a dataset from a list of images
dataset = fo.Dataset.from_images(
    img_paths,
)

# place1_path = '/data0/indic_data/images/places'
place1_path = '/data0/teja_codes/ImmersoAiResearch/ImageMatchingAlgo/image_search/data'
place1_paths = get_image_paths(place1_path)

place1_paths = place1_paths[:10]

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


model_name = "yolov8l-world-torch"
# model_name = "yolov8m-world-torch"
# model_name = "yolov8x-world-torch"

model = foz.load_zoo_model(
    model_name,
    classes=[ "person running on coal"],
)


# model = foz.load_zoo_model(
#     "zero-shot-detection-transformer-torch",
#     name_or_path="IDEA-Research/grounding-dino-tiny",
#     classes=["person walking on coal"],
# )


final_dataset.apply_model(model, label_field="yolo_world_detections")

print(final_dataset)

patches = final_dataset.to_patches("yolo_world_detections")

fob.compute_similarity(
    final_dataset,
    patches_field="yolo_world_detections",
    model="clip-vit-base32-torch",
    brain_key="gt_sim",
    batch_size=128
)

# # # Step 4: Select a query patch by ID (e.g., first patch)
# query_id = patches.first().id
# print(query_id)

# # # Step 5: Find top 15 patches similar to the query patch by sorting dataset view
# similar_patches_view = patches.sort_by_similarity(
#     query_id,
#     k=15,
#     brain_key="gt_sim"
# )

# session = fo.launch_app(view)
session = fo.launch_app(final_dataset)
# session = fo.launch_app(similar_patches_view)

while True:
    pass