# Actor Face Detection Pipeline

Automated pipeline for detecting and tagging actors in movie frames using face detection, embeddings, and similarity search.

## Pipeline Overview

```
Movie Frames
    ↓
1. Face Detection (CPU) → faces_detected.csv
    ↓
2. Embedding Generation (GPU) → embeddings/
    ↓
3. Actor Search (FAISS) → actor_search_results_combined.csv
    ↓
4. Image Generation → annotated_images/
```

After this, the image generation is done, this is given to the vlm for asking annotations to be done.




## Quick Start

### 1. Edit Configuration

Open `config.py` and set:

```python
# Your movie frames directory
MOVIE_FRAMES_DIR = '/path/to/movie/frames'

# Output directory (will be created)
OUTPUT_DIR = 'output'

# Actor embeddings (pickle files from actor faces)
ACTOR_EMBEDDINGS = [
    '/path/to/actor1_face.pkl',
    '/path/to/actor2_face.pkl',
]

# Actor names for display
ACTOR_NAMES = {
    'actor1_face.pkl': 'Actor One',
    'actor2_face.pkl': 'Actor Two',
}

# GPU IDs to use
EMBEDDING_GPU_IDS = [0, 1, 2, 3]
```

### 2. Run Pipeline

```bash
python main.py
```

That's it! The pipeline will:
- Detect all faces in the movie frames
- Generate embeddings for each face
- Search for matching actors using similarity search
- Create annotated images with bounding boxes

## Configuration Options

### Processing Parameters

```python
# Face Detection (CPU)
FACE_DETECTION_BATCH_SIZE = 32
FACE_DETECTION_CPU_PERCENTAGE = 50  # Use 50% of CPUs

# Embedding Generation (GPU)
EMBEDDING_BATCH_SIZE = 64
EMBEDDING_GPU_IDS = [0, 1, 2, 3]  # Which GPUs to use

# Image Generation
SIMILARITY_THRESHOLD = 0.0  # Filter faces by similarity score
BOX_WIDTH = 3
FONT_SIZE = 16
```

### Skipping Steps

If you need to resume the pipeline or skip certain steps:

```python
SKIP_FACE_DETECTION = False      # Set True if faces already detected
SKIP_EMBEDDING_GENERATION = False # Set True if embeddings already exist
SKIP_ACTOR_SEARCH = False        # Set True if search already done
SKIP_IMAGE_GENERATION = False    # Set True to skip visualization
```

## Output Structure

```
output/
├── faces_detected.csv              # All detected faces
├── embeddings/                     # Face embeddings (mirrors input structure)
│   └── shot_001/
│       └── 1/
│           ├── frame_face_0.pkl
│           └── frame_face_1.pkl
├── faiss/
│   ├── stacked_embeddings/         # Stacked embeddings for FAISS
│   ├── index/                      # FAISS index files
│   │   └── knn.index
│   └── results/                    # Search results
│       ├── actor_search_actor1.csv
│       ├── actor_search_actor2.csv
│       └── actor_search_results_combined.csv
└── annotated_images/               # Images with bounding boxes
    └── shot_001/
        └── 1/
            └── frame.jpg
```

## Individual Scripts

You can also run each step individually:

### Face Detection
```bash
python face_detection_ray.py
```

### Embedding Generation
```bash
python embedding_generation_ray_gpu.py
```

### Actor Search
```bash
python actor_search.py
```

### Image Generation
```bash
python image_generation.py
```

## Requirements

- Python 3.8+
- PyTorch with CUDA
- Ray
- Ultralytics (YOLO)
- DeepFace
- FAISS
- PIL/Pillow
- pandas
- numpy

## Tips

1. **GPU Memory**: If you run out of GPU memory, reduce `EMBEDDING_BATCH_SIZE`

2. **CPU Usage**: Adjust `FACE_DETECTION_CPU_PERCENTAGE` based on your system load

3. **Similarity Threshold**: Start with 0.0 to see all matches, then increase (e.g., 0.5) to filter low-confidence matches

4. **Resuming**: Set skip flags in config.py to resume from a specific step

5. **Multiple Movies**: Create separate config files (e.g., `config_movie1.py`, `config_movie2.py`) and import the appropriate one in `main.py`

## Example Config for New Movie

```python
# config.py
MOVIE_FRAMES_DIR = '/data/movies/inception/frames'
OUTPUT_DIR = 'output_inception'

ACTOR_EMBEDDINGS = [
    'actor_embeddings/dicaprio.pkl',
    'actor_embeddings/cotillard.pkl',
]

ACTOR_NAMES = {
    'dicaprio.pkl': 'Leonardo DiCaprio',
    'cotillard.pkl': 'Marion Cotillard',
}

EMBEDDING_GPU_IDS = [0, 1]  # Use 2 GPUs
```

Then just run:
```bash
python main.py
```

