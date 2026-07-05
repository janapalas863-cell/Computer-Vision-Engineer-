
import os
import json
from collections import defaultdict

import cv2
import numpy as np
from skimage.feature import hog
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_distances

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


# CONFIG

IMAGE_FOLDER = "raw_images"     # put your input images here
OUTPUT_FOLDER = "outputs"
FACE_SIZE = (128, 128)
DBSCAN_EPS = 0.27                    # tune this for your dataset
DBSCAN_MIN_SAMPLES = 2

CASCADE_PATH = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"


#  FACE DETECTION

def detect_faces(image_folder, scale_factor=1.1, min_neighbors=5, min_size=(60, 60)):
    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    if face_cascade.empty():
        raise RuntimeError("Failed to load Haar Cascade classifier.")

    results = []
    for img_name in sorted(os.listdir(image_folder)):
        if not img_name.lower().endswith((".jpg", ".jpeg", ".png")):
            continue

        path = os.path.join(image_folder, img_name)
        image = cv2.imread(path)
        if image is None:
            print(f"⚠️  Could not read: {img_name}")
            continue

        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray = cv2.equalizeHist(gray)  # helps with varying lighting conditions

        faces = face_cascade.detectMultiScale(
            gray, scaleFactor=scale_factor, minNeighbors=min_neighbors, minSize=min_size
        )

        if len(faces) == 0:
            print(f"⚠️  No face detected: {img_name}")
            continue

        # Keep the largest detected face
        faces = sorted(faces, key=lambda b: b[2] * b[3], reverse=True)
        results.append((path, image, faces))
        print(f"✅ {img_name}: {len(faces)} face(s) detected")

    return results


#  EMBEDDING GENERATION (HOG features)

def preprocess_face(bgr_image, box, face_size=FACE_SIZE):
    x, y, w, h = box
    margin_x, margin_y = int(0.15 * w), int(0.15 * h)
    x1 = max(0, x - margin_x)
    y1 = max(0, y - margin_y)
    x2 = min(bgr_image.shape[1], x + w + margin_x)
    y2 = min(bgr_image.shape[0], y + h + margin_y)

    face_crop = bgr_image[y1:y2, x1:x2]
    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    resized = cv2.resize(gray, face_size, interpolation=cv2.INTER_AREA)
    return resized


def embed_face(face_gray):
    features = hog(
        face_gray,
        orientations=9,
        pixels_per_cell=(8, 8),
        cells_per_block=(2, 2),
        block_norm="L2-Hys",
        feature_vector=True,
    )
    norm = np.linalg.norm(features)
    if norm > 0:
        features = features / norm
    return features


def get_embeddings(detected_faces):
    encodings, image_paths = [], []
    for path, image, faces in detected_faces:
        box = faces[0]  # largest face
        face_gray = preprocess_face(image, box)
        vector = embed_face(face_gray)
        encodings.append(vector)
        image_paths.append(path)
    return np.array(encodings), image_paths

# : CLUSTERING (DBSCAN)

def cluster_faces(encodings, image_paths, eps=DBSCAN_EPS, min_samples=DBSCAN_MIN_SAMPLES, metric="cosine"):
    clt = DBSCAN(metric=metric, eps=eps, min_samples=min_samples)
    clt.fit(encodings)
    labels = clt.labels_

    clusters = defaultdict(list)
    for path, label, enc in zip(image_paths, labels, encodings):
        clusters[label].append((path, enc))
    return clusters


#  CONFIDENCE SCORING

def _cosine_distance(a, b):
    return 1 - (np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))


def compute_confidence(clusters):
    results = {}
    for label, items in clusters.items():
        if label == -1:
            for path, _ in items:
                results[os.path.basename(path)] = {"cluster": -1, "confidence": 0.0}
            continue

        embs = np.array([e for _, e in items])
        centroid = embs.mean(axis=0)

        for path, enc in items:
            dist = _cosine_distance(enc, centroid)
            confidence = max(0.0, 1 - dist)
            results[os.path.basename(path)] = {
                "cluster": int(label),
                "confidence": round(float(confidence), 3),
            }
    return results


def save_results(results, out_path):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"Saved results -> {out_path}")


# VISUALIZATION

def visualize(results, image_folder, out_dir):
    grouped = defaultdict(list)
    for filename, info in results.items():
        grouped[info["cluster"]].append((filename, info["confidence"]))

    for cluster_id, items in sorted(grouped.items()):
        items.sort()
        fig, axes = plt.subplots(1, len(items), figsize=(3 * len(items), 3.5))
        if len(items) == 1:
            axes = [axes]

        label = f"Cluster {cluster_id}" if cluster_id != -1 else "Unclustered / Noise"
        for ax, (filename, conf) in zip(axes, items):
            img = mpimg.imread(os.path.join(image_folder, filename))
            ax.imshow(img)
            ax.set_title(f"{filename}\nconf: {conf}", fontsize=8)
            ax.axis("off")

        plt.suptitle(label)
        plt.tight_layout()
        out_path = os.path.join(out_dir, f"cluster_{cluster_id}.png")
        plt.savefig(out_path, dpi=120)
        plt.close(fig)
        print(f"Saved {out_path}")



def run():
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)

    print("Step 1: Detecting faces...")
    detected = detect_faces(IMAGE_FOLDER)

    print("\nStep 2: Generating embeddings...")
    encodings, paths = get_embeddings(detected)
    print(f"Generated {len(encodings)} embeddings.")

    print("\nStep 3: Clustering...")
    clusters = cluster_faces(encodings, paths)
    n_clusters = len([k for k in clusters if k != -1])
    n_noise = len(clusters.get(-1, []))
    print(f"Found {n_clusters} cluster(s), {n_noise} noise/unclustered image(s).")

    print("\nStep 4: Scoring confidence...")
    results = compute_confidence(clusters)
    save_results(results, out_path=os.path.join(OUTPUT_FOLDER, "clusters.json"))

    print("\n=== Final Results ===")
    for img, info in sorted(results.items()):
        print(f"{img:20s} -> cluster {info['cluster']:>2}  confidence {info['confidence']}")

    print("\nStep 5: Generating cluster visualizations...")
    visualize(results, IMAGE_FOLDER, OUTPUT_FOLDER)

    return results


if __name__ == "__main__":
    run()