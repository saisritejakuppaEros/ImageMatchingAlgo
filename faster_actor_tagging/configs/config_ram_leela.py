"""
Configuration File for Actor Face Detection Pipeline - Goliyon Ki Raasleela Ram-Leela (2013)
=====================================================

Edit the variables below for your movie and actors.
All paths can be absolute or relative.
"""

# ========== INPUT/OUTPUT PATHS ==========

# Input: Directory containing movie frames (will be searched recursively)
MOVIE_FRAMES_DIR = '/data0/PArun/Quality_checklist/dataset/cropped_detected_shots/Sep/Goliyon Ki Raasleela Ram-Leela'  # UPDATE THIS PATH

# Output: Base directory for all outputs
OUTPUT_DIR = 'output_data/Ram Leela'

# Actor embeddings: List of paths to actor face embedding pickle files
ACTOR_EMBEDDINGS = [
    '/data0/teja_works/dataset_captioning/actors_embeddings/ranveer_face_6.pkl'
]

# Actor name mapping (for display in output images)
# Key should match the pickle filename (with .pkl extension)
ACTOR_NAMES = {
    'ranveer_face_6.pkl': 'Ranveer Singh'
}

# Actor color mapping (for bounding boxes in output images)
# Use RGB tuples: (R, G, B) where each value is 0-255
ACTOR_COLORS = {
    'ranveer_face_6': (0, 191, 255)  # Deep Sky Blue
}

# Text colors for each actor (to ensure good contrast with background)
# Use 'black' or 'white' depending on background color
ACTOR_TEXT_COLORS = {
    'ranveer_face_6': 'black'  # Black text on blue background
}


# ========== MODEL CONFIGURATION ==========

# Face detection model path (YOLO)
FACE_DETECTION_MODEL = 'yolov12m-face.pt'

# Face embedding model (DeepFace)
# Options: VGG-Face, Facenet, Facenet512, OpenFace, DeepFace, DeepID, ArcFace, Dlib, SFace, GhostFaceNet, Buffalo_L
EMBEDDING_MODEL = 'ArcFace'


# ========== PROCESSING PARAMETERS ==========

# Face Detection (GPU-based)
FACE_DETECTION_BATCH_SIZE = 64  # Number of images per batch (higher = better GPU utilization)
FACE_DETECTION_GPU_IDS = [0, 1, 2, 3, 4, 7]  # List of GPU IDs to use (e.g., [0, 1, 2, 3])

# Embedding Generation (CPU-based)
EMBEDDING_BATCH_SIZE = 32  # Number of faces per batch
EMBEDDING_CPU_PERCENTAGE = 50  # Percentage of available CPUs to use (1-100)

# Actor Search
FAISS_TOP_K = None  # Number of top matches to return (None = return all)

# Image Generation
SIMILARITY_THRESHOLD = 0.0  # Only draw faces with similarity >= threshold (0.0 = all faces)
BOX_WIDTH = 3  # Width of bounding box lines
FONT_SIZE = 16  # Font size for labels


# ========== PIPELINE CONTROL ==========

# Skip steps that are already completed (useful for resuming)
SKIP_FACE_DETECTION = False  # Set to True if faces already detected
SKIP_EMBEDDING_GENERATION = False  # Set to True if embeddings already generated
SKIP_ACTOR_SEARCH = False  # Set to True if actor search already done
SKIP_IMAGE_GENERATION = False  # Set to True if you don't want annotated images

