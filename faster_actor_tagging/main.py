"""
Actor Face Detection Pipeline - Main Orchestrator
==================================================

This script runs the complete pipeline:
1. Face Detection (GPU parallelized with Ray)
2. Embedding Generation (CPU parallelized with Ray)
3. Actor Search (FAISS similarity search for each actor)
4. Image Generation (Visualize results with bounding boxes)

Usage:
    1. Edit config.py with your movie and actor paths
    2. Run: python main.py
"""

import os
import sys
import time
from pathlib import Path
import multiprocessing
import pandas as pd

# Import configuration
import config

# Import pipeline modules
from face_detection_ray_gpu import main as face_detection_main
from embedding_generation_ray import main as embedding_generation_main
from actor_search import (
    load_all_frame_embeddings,
    normalize_embeddings,
    save_embeddings_for_faiss,
    build_faiss_index_manual,
    load_actor_embedding,
    search_actor_in_frames,
    create_results_csv
)


def print_header(title):
    """Print a formatted section header."""
    print(f"\n{'='*80}")
    print(f"{title.center(80)}")
    print(f"{'='*80}\n")


def print_step(step_num, total_steps, step_name):
    """Print a formatted step indicator."""
    print(f"\n{'━'*80}")
    print(f"STEP {step_num}/{total_steps}: {step_name}")
    print(f"{'━'*80}\n")


def get_output_paths(output_dir):
    """Generate all output paths from base output directory."""
    output_dir = Path(output_dir)
    
    return {
        'faces_csv': output_dir / 'faces_detected.csv',
        'embeddings_dir': output_dir / 'embeddings',
        'faiss_stacked': output_dir / 'faiss' / 'stacked_embeddings',
        'faiss_index': output_dir / 'faiss' / 'index',
        'faiss_index_file': output_dir / 'faiss' / 'index' / 'knn.index',
        'results_dir': output_dir / 'faiss' / 'results',
        'images_dir': output_dir / 'annotated_images'
    }


def step_1_face_detection(movie_dir, output_paths, model_path, batch_size, gpu_ids):
    """Step 1: Detect faces in all movie frames."""
    print_step(1, 4, "FACE DETECTION")
    
    if config.SKIP_FACE_DETECTION and output_paths['faces_csv'].exists():
        print(f"⏭️  Skipping face detection (SKIP_FACE_DETECTION=True)")
        print(f"   Using existing: {output_paths['faces_csv']}")
        df = pd.read_csv(output_paths['faces_csv'])
        print(f"   Found {len(df)} face detections")
        return
    
    print(f"📷 Detecting faces in: {movie_dir}")
    print(f"📁 Output CSV: {output_paths['faces_csv']}")
    print(f"🤖 Model: {model_path}")
    print(f"📦 Batch size: {batch_size}")
    print(f"🎮 GPUs: {gpu_ids}\n")
    
    # Create output directory
    output_paths['faces_csv'].parent.mkdir(parents=True, exist_ok=True)
    
    # Run face detection
    start_time = time.time()
    face_detection_main(
        input_dir=str(movie_dir),
        output_csv=str(output_paths['faces_csv']),
        batch_size=batch_size,
        model_path=model_path,
        gpu_ids=gpu_ids
    )
    
    elapsed = time.time() - start_time
    print(f"\n✅ Face detection completed in {elapsed:.2f} seconds")


def step_2_embedding_generation(output_paths, movie_dir, model_name, batch_size, cpu_percentage):
    """Step 2: Generate embeddings for all detected faces."""
    print_step(2, 4, "EMBEDDING GENERATION")
    
    if config.SKIP_EMBEDDING_GENERATION and output_paths['embeddings_dir'].exists():
        print(f"⏭️  Skipping embedding generation (SKIP_EMBEDDING_GENERATION=True)")
        print(f"   Using existing: {output_paths['embeddings_dir']}")
        return
    
    print(f"🧠 Generating embeddings for detected faces")
    print(f"📁 Input CSV: {output_paths['faces_csv']}")
    print(f"📂 Output directory: {output_paths['embeddings_dir']}")
    print(f"🤖 Model: {model_name}")
    print(f"📦 Batch size: {batch_size}")
    print(f"💻 CPU usage: {cpu_percentage}%\n")
    
    # Calculate number of workers
    total_cpus = multiprocessing.cpu_count()
    num_workers = max(1, int(total_cpus * (cpu_percentage / 100.0)))
    
    # Run embedding generation
    start_time = time.time()
    embedding_generation_main(
        csv_path=str(output_paths['faces_csv']),
        input_root=str(movie_dir),
        output_root=str(output_paths['embeddings_dir']),
        model_name=model_name,
        batch_size=batch_size,
        num_workers=num_workers
    )
    
    elapsed = time.time() - start_time
    print(f"\n✅ Embedding generation completed in {elapsed:.2f} seconds")


def step_3_actor_search(output_paths, actor_embeddings, top_k):
    """Step 3: Search for each actor in the frame embeddings."""
    print_step(3, 4, "ACTOR SEARCH")
    
    if config.SKIP_ACTOR_SEARCH:
        print(f"⏭️  Skipping actor search (SKIP_ACTOR_SEARCH=True)")
        combined_csv = output_paths['results_dir'] / 'actor_search_results_combined.csv'
        if combined_csv.exists():
            print(f"   Using existing: {combined_csv}")
        return
    
    print(f"🔍 Searching for {len(actor_embeddings)} actors in frame embeddings")
    print(f"📁 Frame embeddings: {output_paths['embeddings_dir']}")
    print(f"📁 Face detections CSV: {output_paths['faces_csv']}")
    print(f"💾 Output directory: {output_paths['results_dir']}\n")
    
    # Create output directory
    output_paths['results_dir'].mkdir(parents=True, exist_ok=True)
    
    start_time = time.time()
    
    # Load and prepare frame embeddings (once for all actors)
    print("📥 Loading frame embeddings...")
    embeddings_array, metadata_list = load_all_frame_embeddings(
        str(output_paths['embeddings_dir']),
        str(output_paths['faces_csv'])
    )
    
    print("🔄 Normalizing embeddings...")
    embeddings_array_norm = normalize_embeddings(embeddings_array)
    
    # Save stacked embeddings
    print("💾 Saving stacked embeddings...")
    save_embeddings_for_faiss(embeddings_array_norm, str(output_paths['faiss_stacked']))
    
    # Build FAISS index (once for all actors)
    print("🏗️  Building FAISS index...")
    index_path = build_faiss_index_manual(embeddings_array_norm, str(output_paths['faiss_index']))
    
    # Determine top_k
    if top_k is None:
        top_k = len(metadata_list)
    
    # Search for each actor
    all_results = []
    for i, actor_path in enumerate(actor_embeddings, 1):
        actor_name = Path(actor_path).stem
        print(f"\n🎭 Searching for actor {i}/{len(actor_embeddings)}: {actor_name}")
        print(f"   Embedding: {actor_path}")
        
        # Load actor embedding
        actor_embedding, actor_data = load_actor_embedding(actor_path)
        actor_data['pkl_path'] = actor_path
        
        # Search
        results = search_actor_in_frames(index_path, actor_embedding, metadata_list, top_k=top_k)
        
        # Save individual CSV for this actor
        output_csv = output_paths['results_dir'] / f'actor_search_{actor_name}.csv'
        create_results_csv(results, actor_data, str(output_csv))
        
        # Collect for combined CSV - add actor information to each result
        for result in results:
            result['actor_name'] = actor_name
            result['actor_embedding_path'] = actor_path
            all_results.append(result)
        
        print(f"   ✅ Found {len(results)} matches")
    
    # Create combined CSV with all actors
    if all_results:
        print(f"\n📊 Creating combined results CSV...")
        combined_csv = output_paths['results_dir'] / 'actor_search_results_combined.csv'
        
        csv_data = []
        for result in all_results:
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
                'actor_name': result['actor_name'],
                'actor_embedding_path': result['actor_embedding_path']
            })
        
        df_combined = pd.DataFrame(csv_data)
        df_combined.to_csv(combined_csv, index=False)
        print(f"   Saved: {combined_csv}")
        print(f"   Total matches: {len(df_combined)}")
    
    elapsed = time.time() - start_time
    print(f"\n✅ Actor search completed in {elapsed:.2f} seconds")


def step_4_image_generation(output_paths, movie_dir):
    """Step 4: Generate annotated images with bounding boxes."""
    print_step(4, 4, "IMAGE GENERATION")
    
    if config.SKIP_IMAGE_GENERATION:
        print(f"⏭️  Skipping image generation (SKIP_IMAGE_GENERATION=True)")
        return
    
    combined_csv = output_paths['results_dir'] / 'actor_search_results_combined.csv'
    
    print(f"🎨 Generating annotated images")
    print(f"📁 Input CSV: {combined_csv}")
    print(f"📂 Output directory: {output_paths['images_dir']}")
    print(f"🎯 Similarity threshold: {config.SIMILARITY_THRESHOLD}\n")
    
    # Create output directory
    output_paths['images_dir'].mkdir(parents=True, exist_ok=True)
    
    # Load CSV
    df = pd.read_csv(combined_csv)
    print(f"📊 Loaded {len(df)} face detections from CSV")
    
    # Filter by similarity threshold
    if config.SIMILARITY_THRESHOLD > 0.0:
        df_filtered = df[df['similarity_score'] >= config.SIMILARITY_THRESHOLD]
        print(f"🔍 Filtered to {len(df_filtered)} faces with similarity >= {config.SIMILARITY_THRESHOLD}")
    else:
        df_filtered = df
        print("📋 No filtering applied (threshold = 0.0)")
    
    if len(df_filtered) == 0:
        print("⚠️  No faces to draw after filtering!")
        return
    
    # Import image generation utilities
    from PIL import Image, ImageDraw, ImageFont
    from tqdm import tqdm
    
    # Load font
    try:
        font = ImageFont.truetype("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf", config.FONT_SIZE)
    except:
        try:
            font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf", config.FONT_SIZE)
        except:
            font = ImageFont.load_default()
            print("⚠️  Could not load TrueType font, using default font")
    
    # Statistics
    print(f"\n📈 Statistics:")
    print(f"   Unique images to process: {df_filtered['frame_image_path'].nunique()}")
    actor_counts = df_filtered['actor_name'].value_counts()
    for actor, count in actor_counts.items():
        display_name = config.ACTOR_NAMES.get(actor + '.pkl', actor)
        print(f"   - {display_name}: {count} faces")
    
    # Process images
    print(f"\n🖼️  Processing images...")
    grouped = df_filtered.groupby('frame_image_path')
    
    start_time = time.time()
    images_processed = 0
    images_with_errors = 0
    images_skipped = 0
    
    for frame_image_path, group_df in tqdm(grouped, total=len(grouped), desc="Processing images", mininterval=0.5):
        image = None
        try:
            # Check if file exists and is readable
            if not os.path.exists(frame_image_path):
                print(f"\n⚠️  File not found: {frame_image_path}")
                images_skipped += 1
                continue
            
            # Check file size (skip if > 50MB to avoid memory issues)
            file_size = os.path.getsize(frame_image_path)
            if file_size > 50 * 1024 * 1024:  # 50MB
                print(f"\n⚠️  Skipping large file ({file_size / (1024*1024):.1f}MB): {frame_image_path}")
                images_skipped += 1
                continue
            
            # Load image with timeout protection
            image = Image.open(frame_image_path)
            
            # Convert to RGB if needed (some images might be RGBA, grayscale, etc.)
            if image.mode not in ('RGB', 'RGBA'):
                image = image.convert('RGB')
            
            draw = ImageDraw.Draw(image)
            
            # Draw all faces in this image
            for _, row in group_df.iterrows():
                try:
                    face_x1 = int(row['face_x1'])
                    face_y1 = int(row['face_y1'])
                    face_x2 = int(row['face_x2'])
                    face_y2 = int(row['face_y2'])
                    actor_name = row['actor_name']
                    similarity_score = row['similarity_score']
                    
                    # Get display name and color
                    actor_pkl_name = os.path.basename(row['actor_embedding_path'])
                    display_name = config.ACTOR_NAMES.get(actor_pkl_name, actor_name)
                    
                    # Get color (RGB tuple) and text color
                    color = config.ACTOR_COLORS.get(actor_name, (255, 255, 0))  # Default: yellow RGB
                    text_color = config.ACTOR_TEXT_COLORS.get(actor_name, 'black')  # Default: black text
                    
                    # Draw bounding box
                    draw.rectangle((face_x1, face_y1, face_x2, face_y2), outline=color, width=config.BOX_WIDTH)
                    
                    # Create label
                    label = f"{display_name} ({similarity_score:.3f})"
                    text_y = max(face_y1 - 25, 5)
                    
                    # Draw text background
                    try:
                        bbox = draw.textbbox((face_x1, text_y), label, font=font)
                        bbox = (bbox[0] - 2, bbox[1] - 2, bbox[2] + 2, bbox[3] + 2)
                        draw.rectangle(bbox, fill=color)
                    except:
                        draw.rectangle((face_x1, text_y, face_x1 + 200, text_y + 20), fill=color)
                    
                    # Draw label with appropriate text color
                    draw.text((face_x1, text_y), label, fill=text_color, font=font)
                    
                except Exception as e:
                    # Skip this face if there's an error drawing it
                    continue
            
            # Save image (preserve directory structure)
            path_parts = frame_image_path.split('/')
            shot_idx = None
            for i, part in enumerate(path_parts):
                if part.startswith('shot_'):
                    shot_idx = i
                    break
            
            if shot_idx is not None:
                relative_path = '/'.join(path_parts[shot_idx:])
            else:
                relative_path = '/'.join(path_parts[-3:])
            
            save_path = output_paths['images_dir'] / relative_path
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save with quality settings to avoid hanging
            image.save(save_path, quality=95, optimize=False)
            images_processed += 1
            
        except KeyboardInterrupt:
            print(f"\n\n⚠️  Interrupted by user!")
            if image:
                image.close()
            raise
            
        except Exception as e:
            print(f"\n❌ Error processing {frame_image_path}: {e}")
            images_with_errors += 1
            
        finally:
            # Always close the image to free resources
            if image:
                try:
                    image.close()
                except:
                    pass
    
    elapsed = time.time() - start_time
    print(f"\n✅ Image generation completed in {elapsed:.2f} seconds")
    print(f"   Images processed: {images_processed}")
    print(f"   Images skipped: {images_skipped}")
    print(f"   Images with errors: {images_with_errors}")
    print(f"   Total faces drawn: {len(df_filtered)}")


def main():
    """Main pipeline orchestrator."""
    print_header("ACTOR FACE DETECTION PIPELINE")
    
    # Display configuration
    print("📋 Configuration:")
    print(f"   Movie frames: {config.MOVIE_FRAMES_DIR}")
    print(f"   Output directory: {config.OUTPUT_DIR}")
    print(f"   Number of actors: {len(config.ACTOR_EMBEDDINGS)}")
    print(f"   Face detection model: {config.FACE_DETECTION_MODEL}")
    print(f"   Face detection GPUs: {config.FACE_DETECTION_GPU_IDS}")
    print(f"   Embedding model: {config.EMBEDDING_MODEL}")
    print(f"   Embedding CPU usage: {config.EMBEDDING_CPU_PERCENTAGE}%")
    
    # Get output paths
    output_paths = get_output_paths(config.OUTPUT_DIR)
    
    # Pipeline start time
    pipeline_start = time.time()
    
    try:
        # Step 1: Face Detection
        step_1_face_detection(
            movie_dir=config.MOVIE_FRAMES_DIR,
            output_paths=output_paths,
            model_path=config.FACE_DETECTION_MODEL,
            batch_size=config.FACE_DETECTION_BATCH_SIZE,
            gpu_ids=config.FACE_DETECTION_GPU_IDS
        )
        
        # Step 2: Embedding Generation
        step_2_embedding_generation(
            output_paths=output_paths,
            movie_dir=config.MOVIE_FRAMES_DIR,
            model_name=config.EMBEDDING_MODEL,
            batch_size=config.EMBEDDING_BATCH_SIZE,
            cpu_percentage=config.EMBEDDING_CPU_PERCENTAGE
        )
        
        # Step 3: Actor Search
        step_3_actor_search(
            output_paths=output_paths,
            actor_embeddings=config.ACTOR_EMBEDDINGS,
            top_k=config.FAISS_TOP_K
        )
        
        # Step 4: Image Generation
        step_4_image_generation(
            output_paths=output_paths,
            movie_dir=config.MOVIE_FRAMES_DIR
        )
        
        # Pipeline complete
        total_elapsed = time.time() - pipeline_start
        
        print_header("PIPELINE COMPLETE!")
        print(f"⏱️  Total time: {total_elapsed:.2f} seconds ({total_elapsed/60:.2f} minutes)")
        print(f"\n📁 Output files:")
        print(f"   Face detections: {output_paths['faces_csv']}")
        print(f"   Embeddings: {output_paths['embeddings_dir']}")
        print(f"   FAISS index: {output_paths['faiss_index_file']}")
        print(f"   Search results: {output_paths['results_dir']}")
        print(f"   Annotated images: {output_paths['images_dir']}")
        print(f"\n{'='*80}\n")
        
    except Exception as e:
        print(f"\n❌ Pipeline failed with error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

