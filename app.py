import os
import random
from flask import Flask, render_template, request, send_from_directory
import numpy as np
import trimesh
from scipy.fft import fftn
from pymongo import MongoClient
from sklearn.decomposition import PCA
from scipy.special import sph_harm
from scipy.spatial.distance import euclidean
from scipy.ndimage import zoom
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
if not os.path.exists(app.config['UPLOAD_FOLDER']):
    os.makedirs(app.config['UPLOAD_FOLDER'])

# Define the new assets path
STATIC_FOLDER = "static/thumbnails"
app.config['STATIC_FOLDER'] = STATIC_FOLDER

# Function to normalize a mesh
def normalize_mesh(mesh):
    centroid = mesh.vertices.mean(axis=0)
    mesh.vertices -= centroid
    max_distance = np.max(np.linalg.norm(mesh.vertices, axis=1))
    mesh.vertices /= max_distance
    pca = PCA(n_components=3)
    mesh.vertices = pca.fit_transform(mesh.vertices)
    return mesh

# Function to voxelize a mesh
def voxelize_mesh(mesh, resolution=64):
    voxels = mesh.voxelized(pitch=2/resolution).fill()
    return voxels.matrix

# Function to compute Fourier descriptor
def compute_fourier_descriptor(voxels):
    fourier_coeffs = fftn(voxels)
    return np.abs(fourier_coeffs).flatten()

# Function to compute Zernike descriptor
def compute_zernike_descriptor_3d(voxels, degree=8):
    indices = np.argwhere(voxels > 0)
    x, y, z = indices[:, 0], indices[:, 1], indices[:, 2]
    rho = np.sqrt(x**2 + y**2 + z**2)
    theta = np.arctan2(y, x)
    phi = np.arccos(z / (rho + 1e-10))
    zernike_descriptor = []
    for n in range(degree + 1):
        for m in range(-n, n + 1):
            zernike_poly = sph_harm(m, n, theta, phi) * np.sqrt((2 * n + 1) / (4 * np.pi))
            moment = np.sum(zernike_poly * voxels[x, y, z])
            zernike_descriptor.append(moment)
    return np.real(np.array(zernike_descriptor))

# Function to normalize a descriptor
def normalize_descriptor(descriptor):
    min_val = np.min(descriptor)
    max_val = np.max(descriptor)
    return (descriptor - min_val) / (max_val - min_val + 1e-10)

# Function to resize a descriptor
def resize_descriptor(descriptor, target_shape):
    factors = [target_shape[i] / descriptor.shape[i] for i in range(len(descriptor.shape))]
    return zoom(descriptor, factors)

# Function to get descriptors from MongoDB
def get_descriptors_from_mongodb():
    client = MongoClient("mongodb://localhost:27017/")
    db = client["3d_models"]
    collection = db["descriptors2"]
    descriptors = list(collection.find({}, {"_id": 0, "file_name": 1, "fourier_descriptor": 1, "zernike_descriptor": 1}))
    return descriptors

# Function to compute similarity between descriptors
def compute_similarity(descriptor1, descriptor2):
    if descriptor1.shape != descriptor2.shape:
        raise ValueError(f"Descriptors have different shapes: {descriptor1.shape} and {descriptor2.shape}")
    return euclidean(descriptor1, descriptor2)

def find_thumbnail(thumbnails_directory, model_name):
    model_name_lower = model_name.lower().replace(" ", "_")
    for root, _, files in os.walk(thumbnails_directory):
        for file in files:
            file_lower = file.lower().replace(" ", "_")
            if file_lower.startswith(model_name_lower) and file_lower.endswith(".jpg"):
                # Return the path relative to STATIC_FOLDER
                relative_path = os.path.relpath(os.path.join(root, file), thumbnails_directory)
                return relative_path
    return None



# Function to find similar models
def find_similar_models(query_file_path, descriptors_from_db, thumbnails_directory, top_k=10):
    mesh = trimesh.load(query_file_path)
    mesh = normalize_mesh(mesh)
    resolution = 64
    voxels = voxelize_mesh(mesh, resolution=resolution)
    fourier_descriptor_query = compute_fourier_descriptor(voxels)
    zernike_descriptor_query = compute_zernike_descriptor_3d(voxels)
    fourier_descriptor_query = normalize_descriptor(fourier_descriptor_query)
    zernike_descriptor_query = normalize_descriptor(zernike_descriptor_query)
    similarity_scores = []
    for descriptor in descriptors_from_db:
        fourier_descriptor_db = np.array(descriptor["fourier_descriptor"])
        zernike_descriptor_db = np.array(descriptor["zernike_descriptor"])
        if fourier_descriptor_query.shape != fourier_descriptor_db.shape:
            fourier_descriptor_query = resize_descriptor(fourier_descriptor_query, fourier_descriptor_db.shape)
        if zernike_descriptor_query.shape != zernike_descriptor_db.shape:
            zernike_descriptor_query = resize_descriptor(zernike_descriptor_query, zernike_descriptor_db.shape)
        fourier_similarity = compute_similarity(fourier_descriptor_query, fourier_descriptor_db)
        zernike_similarity = compute_similarity(zernike_descriptor_query, zernike_descriptor_db)
        combined_similarity = 0.5 * fourier_similarity + 0.5 * zernike_similarity
        max_score = np.sqrt(len(fourier_descriptor_query))
        similarity_percentage = 100 * (1 - combined_similarity / max_score)
        model_name = descriptor["file_name"].replace(".obj", "")
        similarity_scores.append((model_name, similarity_percentage))
    similarity_scores.sort(key=lambda x: x[1], reverse=True)
    return similarity_scores[:top_k]

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'file' not in request.files:
            return render_template('index.html', error='No file part')
        file = request.files['file']
        if file.filename == '':
            return render_template('index.html', error='No selected file')
        if file and allowed_file(file.filename):
            filename = file.filename
            file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
            file.save(file_path)
            try:
                descriptors_from_db = get_descriptors_from_mongodb()
                similar_models = find_similar_models(file_path, descriptors_from_db, app.config['STATIC_FOLDER'], top_k=10)
                thumbnails = []
                for model_name, score in similar_models:
                    thumb_path = find_thumbnail(app.config['STATIC_FOLDER'], model_name)
                    if thumb_path:
                        thumbnails.append((model_name, thumb_path, score))
                return render_template('results.html', query_model=filename, thumbnails=thumbnails)
            except Exception as e:
                return render_template('index.html', error=str(e))
    return render_template('index.html')

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() == 'obj'

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/static/thumbnails/<path:filename>')
def serve_thumbnail(filename):
    print(f"Thumbnail requested: {filename}")
    return send_from_directory(app.config['STATIC_FOLDER'], filename)

if __name__ == '__main__':
    app.run(debug=True)