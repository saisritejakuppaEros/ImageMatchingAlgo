# Movie Configuration Files - Usage Guide

This directory contains pre-configured config files for multiple movies with male actors only.

## Available Configurations

### Shah Rukh Khan Movies
1. **config_ra_one.py** - Ra One (2011)
2. **config_hey_ram.py** - Hey Ram (2000) - Note: Only SRK embedding available
3. **config_hum_tumhare_hain_sanam.py** - Hum Tumhare Hain Sanam (2002)
4. **config.py** - Devdas (2002) - Default config (includes female actors)

### Ranbir Kapoor Movies
5. **config_rockstar.py** - Rockstar (2011)
6. **config_anjaana_anjaani.py** - Anjaana Anjaani (2010)

### R. Madhavan Movies
7. **config_aarya.py** - Aarya (2007)
8. **config_tanu_weds_manu_returns.py** - Tanu Weds Manu Returns (2015)

### Dhanush Movies
9. **config_3moonu.py** - 3 Moonu (2012)
10. **config_raanjhanaa.py** - Raanjhanaa (2013)

### Ranveer Singh Movies
11. **config_bajirao_mastani.py** - Bajirao Mastani (2015)
12. **config_ram_leela.py** - Goliyon Ki Raasleela Ram-Leela (2013)

## How to Use

### Step 1: Update the Movie Frames Directory
Open the config file for your movie and update the `MOVIE_FRAMES_DIR` path:

```python
MOVIE_FRAMES_DIR = '/path/to/your/movie/frames'  # UPDATE THIS PATH
```

### Step 2: Run the Pipeline

#### Method 1: Import the config directly
```bash
python main.py --config config_rockstar
```

#### Method 2: Copy and rename to config.py
```bash
cp config_rockstar.py config.py
python main.py
```

#### Method 3: Modify main.py to import specific config
Edit main.py line 24:
```python
# Change from:
import config

# To:
import config_rockstar as config
```
Then run:
```bash
python main.py
```

## Actor Embeddings Used

All configs use embeddings from: `/data0/teja_works/dataset_captioning/actors_embeddings/`

- **Ranbir Kapoor**: `ranbir_face_3.pkl`
- **Shah Rukh Khan**: `srk_face_2.pkl`
- **R. Madhavan**: `madhavan_face_4.pkl`
- **Dhanush**: `danush_face_5.pkl`
- **Ranveer Singh**: `ranveer_face_6.pkl`

## Color Scheme

Each actor has been assigned a unique color:
- **Ranbir Kapoor**: Red (255, 50, 50)
- **Shah Rukh Khan**: Green (50, 255, 50)
- **R. Madhavan**: Orange (255, 165, 0)
- **Dhanush**: Blue Violet (138, 43, 226)
- **Ranveer Singh**: Deep Sky Blue (0, 191, 255)

## Configuration Parameters

All configs include standard parameters:
- **Face Detection**: Batch size 64, GPUs [0,1,2,3,4,7]
- **Embedding**: ArcFace model, CPU 50%, Batch size 32
- **FAISS**: Top-K = None (returns all matches)
- **Visualization**: Threshold 0.0, Box width 3, Font size 16

## Quick Reference: Run Pipeline for a Movie

```bash
# Example: Process Rockstar movie
cd /data0/teja_works/dataset_captioning/faster_actor_tagging

# Edit the config
nano config_rockstar.py
# Update MOVIE_FRAMES_DIR to your frames directory

# Copy to main config
cp config_rockstar.py config.py

# Run the pipeline
python main.py
```

## Output Structure

For each movie, output will be saved in the directory specified by `OUTPUT_DIR`:
```
{movie_name}_output/
├── faces_detected.csv
├── embeddings/
├── faiss/
│   ├── stacked_embeddings/
│   ├── index/
│   └── results/
└── annotated_images/
```

## Notes

- All paths can be absolute or relative
- Female actor embeddings are excluded (male actors only)
- GPU IDs can be adjusted based on your system
- Similarity threshold can be adjusted for stricter matching

