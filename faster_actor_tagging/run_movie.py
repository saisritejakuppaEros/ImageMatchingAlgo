#!/usr/bin/env python
"""
Simple Movie Runner Script
===========================

Usage:
    python run_movie.py <movie_name>

Example:
    python run_movie.py rockstar
    python run_movie.py devdas
    python run_movie.py bajirao_mastani

This script temporarily uses the specified movie config and runs the pipeline.
"""

import sys
import importlib
import os

# Available movie configurations
AVAILABLE_CONFIGS = {
    # 'rockstar': 'configs.config_rockstar',
    # 'anjaana_anjaani': 'configs.config_anjaana_anjaani',
    # 'ra_one': 'configs.config_ra_one',
    # 'devdas': 'configs.config_devadas',  # Default config
    # 'hey_ram': 'configs.config_hey_ram',
    # 'hum_tumhare_hain_sanam': 'configs.config_hum_tumhare_hain_sanam',
    # 'aarya': 'configs.config_aarya',
    # 'tanu_weds_manu_returns': 'configs.config_tanu_weds_manu_returns',
    # '3moonu': 'configs.config_3moonu',
    'raanjhanaa': 'configs.config_raanjhanaa',
    # 'bajirao_mastani': 'configs.config_bajirao_mastani',
    # 'ram_leela': 'configs.config_ram_leela'
}

def print_usage():
    """Print usage information."""
    print("=" * 80)
    print("Movie Actor Tagging Runner".center(80))
    print("=" * 80)
    print("\nUsage: python run_movie.py <movie_name>\n")
    print("Available movies:")
    for i, (movie, _) in enumerate(AVAILABLE_CONFIGS.items(), 1):
        print(f"  {i:2d}. {movie}")
    print("\nExample:")
    print("  python run_movie.py rockstar")
    print("=" * 80)

def run_single_movie(movie_name, config_module):
    """Run pipeline for a single movie."""
    print("\n" + "=" * 80)
    print(f"Running pipeline for: {movie_name.upper()}".center(80))
    print(f"Using config: {config_module}.py".center(80))
    print("=" * 80)
    
    # Dynamically import the config module
    try:
        config = importlib.import_module(config_module)
    except ModuleNotFoundError:
        print(f"❌ Error: Config file '{config_module}.py' not found!")
        return False
    
    # Verify movie frames directory exists
    if not os.path.exists(config.MOVIE_FRAMES_DIR):
        print(f"\n⚠️  WARNING: Movie frames directory not found!")
        print(f"   Path: {config.MOVIE_FRAMES_DIR}")
        print(f"   Skipping {movie_name}...\n")
        return False
    
    # Display configuration
    print(f"\n📋 Configuration:")
    print(f"   Movie frames: {config.MOVIE_FRAMES_DIR}")
    print(f"   Output directory: {config.OUTPUT_DIR}")
    print(f"   Number of actors: {len(config.ACTOR_EMBEDDINGS)}")
    print(f"   Actors:")
    for pkl_file in config.ACTOR_EMBEDDINGS:
        actor_name = config.ACTOR_NAMES.get(os.path.basename(pkl_file), 'Unknown')
        print(f"      - {actor_name}")
    
    print(f"\n   Face detection model: {config.FACE_DETECTION_MODEL}")
    print(f"   Face detection GPUs: {config.FACE_DETECTION_GPU_IDS}")
    print(f"   Embedding model: {config.EMBEDDING_MODEL}")
    
    # Import and inject config into main module
    import main as main_module
    main_module.config = config
    
    # Run the pipeline
    print("\n🚀 Starting pipeline...\n")
    try:
        main_module.main()
        print(f"\n✅ {movie_name.upper()} completed successfully!\n")
        return True
    except Exception as e:
        print(f"\n❌ {movie_name.upper()} failed with error: {e}\n")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Main runner function - runs all active movies in AVAILABLE_CONFIGS."""
    print("=" * 80)
    print("Movie Actor Tagging - Batch Runner".center(80))
    print("=" * 80)
    print(f"\nFound {len(AVAILABLE_CONFIGS)} active movie(s) to process:")
    for i, movie in enumerate(AVAILABLE_CONFIGS.keys(), 1):
        print(f"  {i}. {movie}")
    print()
    
    results = {}
    for movie_name, config_module in AVAILABLE_CONFIGS.items():
        success = run_single_movie(movie_name, config_module)
        results[movie_name] = success
    
    # Final summary
    print("\n" + "=" * 80)
    print("BATCH PROCESSING COMPLETE".center(80))
    print("=" * 80)
    print("\nResults Summary:")
    for movie_name, success in results.items():
        status = "✅ SUCCESS" if success else "❌ FAILED/SKIPPED"
        print(f"  {movie_name:30s} - {status}")
    print("\n" + "=" * 80)

if __name__ == "__main__":
    main()

