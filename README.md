# 3D Model Similarity Finder

This project is a Flask-based web application that allows users to upload 3D `.obj` files and find the most similar models from a pre-existing database using Fourier and Zernike descriptors.

## Features

- Upload `.obj` files to query the database.
- Normalize and voxelize 3D models for descriptor computation.
- Compute Fourier and Zernike descriptors to compare 3D models.
- Retrieve the top 10 most similar models from the database.
- Display thumbnails of similar models with their similarity percentages.

## Directory Structure

```
miniprojet2
├─ app.py                       # Main application file
├─ static
│  ├─ styles.css               # CSS styles (optional, currently unused)
│  └─ thumbnails               # Thumbnails for database models
│     ├─ 00110054.jpg
│     ├─ 08 dune series trash receptacle ds-tr.jpg
│     ├─ ...
├─ templates
│  ├─ index.html               # Upload page
│  └─ results.html             # Results page displaying similar models
└─ uploads
   ├─ Abstractshape1.obj       # Sample uploaded 3D models
   ├─ Abstractshape11.obj
   ├─ ...
```

## Prerequisites

Make sure you have the following installed:

- Python 3.8+
- Flask
- NumPy
- SciPy
- scikit-learn
- trimesh
- matplotlib
- PyMongo (for MongoDB integration)
- MongoDB (running locally or remotely)

## Installation

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/miniprojet2.git
   cd miniprojet2
   ```

2. Create a virtual environment and install dependencies:
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   pip install -r requirements.txt
   ```

3. Start MongoDB and ensure a database named `3d_models` exists with a collection `descriptors2` containing precomputed descriptors for models.

## Usage

1. Run the Flask application:
   ```bash
   python app.py
   ```

2. Open your browser and navigate to:
   ```
   http://127.0.0.1:5000/
   ```

3. Upload a `.obj` file, and the application will:
   - Normalize and voxelize the 3D model.
   - Compute Fourier and Zernike descriptors.
   - Compare the descriptors against the database.
   - Display the top 10 similar models with their thumbnails and similarity percentages.

## Configuration

- **Upload Folder**: `app.config['UPLOAD_FOLDER']` specifies where uploaded files are stored (`uploads/` directory).
- **Static Thumbnails**: Thumbnails for database models are located in `static/thumbnails/`.

## Key Functions

### `normalize_mesh(mesh)`
Normalizes a mesh by centering it and scaling it to fit within a unit sphere.

### `voxelize_mesh(mesh, resolution=64)`
Converts the mesh into a 3D voxel grid with the specified resolution.

### `compute_fourier_descriptor(voxels)`
Computes the Fourier descriptor from the voxel grid.

### `compute_zernike_descriptor_3d(voxels, degree=8)`
Computes the Zernike descriptor up to the specified degree.

### `find_similar_models(query_file_path, descriptors_from_db, thumbnails_directory, top_k=10)`
Finds the top `k` similar models in the database and returns their names, similarity scores, and thumbnail paths.

## Dependencies

- Flask: Web framework
- NumPy: Numerical computations
- SciPy: Fourier transform and Zernike descriptor calculations
- scikit-learn: PCA for mesh normalization
- trimesh: 3D model handling
- matplotlib: Optional for debugging or visualization
- PyMongo: MongoDB integration

## Database

The MongoDB database should have the following schema for the `descriptors2` collection:

```json
{
  "file_name": "model_name.obj",
  "fourier_descriptor": [ ... ],
  "zernike_descriptor": [ ... ]
}
```
