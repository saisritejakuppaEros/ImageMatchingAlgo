Here’s a cleaner and more professional version of your README instructions:

---

## Actor-Face Search Pipeline

This pipeline allows you to perform actor-face matching by using scraped actor images and movie frames. It includes steps for embedding generation, face search, and CSV output generation.

### 📦 Prerequisites

To use this pipeline, you will need:

* **Actor images**
* **Movie frame images**

---

### 🖼️ Get Actor Images

Use the [Image Scraping Tool](https://github.com/Immerso-AIIP-Ltd/Image_Scraping) to download actor images.
Simply provide the actor names as input to the scraper to fetch a set of representative images.

---

### 🎬 Get Movie Frames

Obtain the movie frames using the Eros database pipeline:

🔗 [Movie Dataset Generation Code](https://github.com/Immerso-AIIP-Ltd/MovieDatasetGeneration/tree/v1.0/Pipeline)

This pipeline provides the frames required for matching with actor images.

---

### ⚙️ Pipeline Steps

1. **Generate Embeddings**
   Use a face embedding model to generate embeddings for both:

   * Actor images
   * Movie frame faces

2. **Perform Face Search**
   Apply a nearest neighbor or similarity search algorithm to match actor embeddings with movie frame embeddings.

3. **Generate Output**
   The script will output:

   * A CSV containing matched actor-frame pairs
   * Corresponding face crops or references (if needed)

---

### ✅ Output Format

The result includes:

* Actor name
* Frame name or ID
* Matching confidence
* (Optional) Face image or coordinates

---

Let me know if you want it converted into a `README.md` file or need badges, usage examples, or install instructions added.
