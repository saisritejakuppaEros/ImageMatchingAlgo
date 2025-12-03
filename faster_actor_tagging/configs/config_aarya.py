"""
Configuration File for Actor Face Detection Pipeline - Aarya (2007)
=====================================================

Edit the variables below for your movie and actors.
All paths can be absolute or relative.
"""

# ========== INPUT/OUTPUT PATHS ==========

# Input: Directory containing movie frames (will be searched recursively)
MOVIE_FRAMES_DIR = '/data0/PArun/Quality_checklist/dataset/cropped_detected_shots/Nov_20/Aarya'  # UPDATE THIS PATH

# Output: Base directory for all outputs
OUTPUT_DIR = 'output_data/Aarya'

# Actor embeddings: List of paths to actor face embedding pickle files
ACTOR_EMBEDDINGS = [
    '/data0/teja_works/dataset_captioning/actors_embeddings/madhavan_face_4.pkl'
]

# Actor name mapping (for display in output images)
# Key should match the pickle filename (with .pkl extension)
ACTOR_NAMES = {
    'madhavan_face_4.pkl': 'R. Madhavan'
}

# Actor color mapping (for bounding boxes in output images)
# Use RGB tuples: (R, G, B) where each value is 0-255
ACTOR_COLORS = {
    'madhavan_face_4': (255, 165, 0)  # Orange
}

# Text colors for each actor (to ensure good contrast with background)
# Use 'black' or 'white' depending on background color
ACTOR_TEXT_COLORS = {
    'madhavan_face_4': 'black'  # Black text on orange background
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

