"""
skin_disease_rf.py

Improved Skin Disease Classification (Machine Learning only)
- Features: HOG (reduced), LBP, GLCM, Color mean/std
- Model: RandomForestClassifier (class_weight="balanced")
- Robust dataset loading and error handling
- Menu loop with Train / Webcam / File / Exit
"""

import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt
from time import time

from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay
from sklearn.model_selection import train_test_split

# -------------------- CONFIG --------------------
DATA_PATH = "SKID"                 # dataset folder with subfolders per class
MODEL_PATH = "rf_model.pkl"
CLASSES_PATH = "classes.txt"

IMAGE_SIZE = (200, 200)
LBP_POINTS = 24
LBP_RADIUS = 3
VALID_EXT = (".jpg", ".jpeg", ".png", ".bmp", ".jfif")

# HOG settings (reduced size for memory/speed)
HOG_ORIENTATIONS = 9
HOG_PIXELS_PER_CELL = (32, 32)     # larger cell -> smaller feature vector
HOG_CELLS_PER_BLOCK = (1, 1)
HOG_BLOCK_NORM = "L2-Hys"

# RandomForest settings
RF_N_ESTIMATORS = 300
RF_N_JOBS = -1   # use all cores

# -------------------- UTILITIES --------------------

def is_image_file(filename):
    return filename.lower().endswith(VALID_EXT)

def preprocess_image(image):
    """Simple preprocessing: blur + normalize intensity"""
    try:
        image = cv2.GaussianBlur(image, (5, 5), 0)
        image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
        return image
    except Exception:
        return image  # if something odd happens, return original

# -------------------- FEATURE EXTRACTORS --------------------

def extract_color_stats(image):
    """Return mean and std for BGR channels (shape: 6)"""
    try:
        pixels = image.reshape(-1, 3).astype(np.float32)
        mean = pixels.mean(axis=0)       # B, G, R
        std = pixels.std(axis=0)
        return np.hstack([mean, std])
    except Exception:
        return np.zeros(6)

def extract_lbp_features(gray):
    try:
        lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
        (hist, _) = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
        hist = hist.astype("float")
        hist /= (hist.sum() + 1e-6)
        return hist
    except Exception:
        return np.zeros(256)

def extract_hog_features(gray):
    try:
        feats = hog(gray,
                    orientations=HOG_ORIENTATIONS,
                    pixels_per_cell=HOG_PIXELS_PER_CELL,
                    cells_per_block=HOG_CELLS_PER_BLOCK,
                    block_norm=HOG_BLOCK_NORM)
        return feats
    except Exception:
        # return a zeros vector of expected approximate size if HOG fails
        # approximate length: ((IMAGE_SIZE//pixels_per_cell)^2)*orientations
        # fallback length computed conservatively:
        approx_len = max(100, (IMAGE_SIZE[0] // HOG_PIXELS_PER_CELL[0]) *
                        (IMAGE_SIZE[1] // HOG_PIXELS_PER_CELL[1]) * HOG_ORIENTATIONS)
        return np.zeros(approx_len)

def extract_glcm_features(gray):
    try:
        glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        props = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation']
        return np.array([graycoprops(glcm, p)[0, 0] for p in props])
    except Exception:
        return np.zeros(5)

def extract_features(image_path):
    """Combined feature vector for a single image path."""
    try:
        img = cv2.imread(image_path)
        if img is None:
            print(f"  [WARN] Cannot read image: {image_path}")
            return None

        img = cv2.resize(img, IMAGE_SIZE)
        img = preprocess_image(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        f_color = extract_color_stats(img)         # 6
        f_hog = extract_hog_features(gray)         # ~small
        f_lbp = extract_lbp_features(gray)         # 256
        f_glcm = extract_glcm_features(gray)       # 5

        features = np.hstack([f_hog, f_lbp, f_glcm, f_color])
        return features

    except Exception as e:
        print(f"  [ERROR] Feature extraction failed for {image_path}: {e}")
        return None

# -------------------- DATA LOADING --------------------

def load_data(limit_per_class=None, verbose=True):
    """
    Load dataset from DATA_PATH.
    - limit_per_class: if set, limit number of images per class (useful for debugging)
    Returns: X (numpy array), y (numpy array), class_names (list)
    """
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"Dataset folder '{DATA_PATH}' not found.")

    class_dirs = sorted([d for d in os.listdir(DATA_PATH) if os.path.isdir(os.path.join(DATA_PATH, d))])
    X, y, class_names = [], [], []

    if verbose:
        print(f"\n[INFO] Found {len(class_dirs)} classes in '{DATA_PATH}'.")

    for label, class_name in enumerate(class_dirs):
        folder = os.path.join(DATA_PATH, class_name)
        files = sorted(os.listdir(folder))
        if verbose:
            print(f"\n[INFO] Processing class {label+1}/{len(class_dirs)}: '{class_name}'  (files: {len(files)})")

        added = 0
        for fname in files:
            if not is_image_file(fname):
                continue
            if limit_per_class and added >= limit_per_class:
                break

            path = os.path.join(folder, fname)
            feats = extract_features(path)
            if feats is None:
                continue

            X.append(feats)
            y.append(label)
            added += 1

            if verbose and (added % 200 == 0):
                print(f"  processed {added} images for class '{class_name}'...")

        class_names.append(class_name)
        if verbose:
            print(f"  -> added {added} images for '{class_name}'")

    X = np.array(X)
    y = np.array(y)
    if verbose:
        print(f"\n[INFO] Finished loading. Total images: {len(X)}")
    return X, y, class_names

# -------------------- TRAIN / EVAL --------------------

def train_random_forest(limit_per_class=None):
    """
    Train RandomForest with balanced class weights.
    limit_per_class: optional integer to limit images per class (for debugging)
    """
    t0 = time()
    X, y, class_names = load_data(limit_per_class=limit_per_class, verbose=True)

    if len(X) == 0:
        print("[ERROR] No images loaded. Check your DATA_PATH and image files.")
        return

    # train-test split with stratify to keep class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )

    print(f"\n[INFO] Train shape: {X_train.shape}, Test shape: {X_test.shape}")

    # RandomForest (handles unscaled features)
    model = RandomForestClassifier(
        n_estimators=RF_N_ESTIMATORS,
        class_weight="balanced",
        n_jobs=RF_N_JOBS,
        random_state=42,
    )

    print("\n[INFO] Training RandomForest (this can take a while)...")
    model.fit(X_train, y_train)

    print("[INFO] Predicting on test set...")
    y_pred = model.predict(X_test)

    acc = accuracy_score(y_test, y_pred)
    print(f"\n🎯 Accuracy on test set: {acc * 100:.2f}%\n")

    print("Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    # Confusion matrix
    ConfusionMatrixDisplay.from_predictions(y_test, y_pred, display_labels=class_names)
    plt.title("Confusion Matrix")
    plt.show()

    # Save model and classes
    with open(MODEL_PATH, "wb") as f:
        pickle.dump(model, f)
    with open(CLASSES_PATH, "w", encoding="utf-8") as f:
        f.write("\n".join(class_names))

    print(f"\n[INFO] Model saved to '{MODEL_PATH}'. Classes saved to '{CLASSES_PATH}'.")
    print(f"[INFO] Training finished in {time() - t0:.1f}s")

# -------------------- PREDICTION --------------------

def predict_image(path):
    """Load model and predict label for a single image file path."""
    if not os.path.exists(MODEL_PATH):
        print("[ERROR] No trained model found. Train the model first.")
        return None

    try:
        with open(MODEL_PATH, "rb") as f:
            model = pickle.load(f)
    except Exception as e:
        print("[ERROR] Failed to load model:", e)
        return None

    try:
        with open(CLASSES_PATH, "r", encoding="utf-8") as f:
            classes = [line.strip() for line in f.readlines()]
    except Exception:
        classes = None

    feats = extract_features(path)
    if feats is None:
        print("[ERROR] Could not extract features from image.")
        return None

    pred_idx = model.predict([feats])[0]
    label = classes[pred_idx] if classes else str(pred_idx)
    return label

# -------------------- WEBCAM CAPTURE --------------------

def capture_from_webcam(filename="capture.jpg"):
    """Open webcam and capture an image when user presses 's'."""
    cam = cv2.VideoCapture(0)
    if not cam.isOpened():
        print("[ERROR] Webcam not accessible.")
        return None

    print("\n[INFO] Webcam opened. Press 's' to save photo, 'q' to cancel.")
    captured = None
    while True:
        ret, frame = cam.read()
        if not ret:
            print("[ERROR] Failed to read frame from webcam.")
            break
        cv2.imshow("Webcam - press 's' to save, 'q' to quit", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("s"):
            cv2.imwrite(filename, frame)
            print("[INFO] Saved capture to", filename)
            captured = filename
            break
        elif key == ord("q"):
            break

    cam.release()
    cv2.destroyAllWindows()
    return captured

# -------------------- MENU --------------------

def menu_loop():
    while True:
        print("""
===============================
  SKIN DISEASE DETECTION (RF)
===============================
1. Train model
2. Predict from webcam
3. Predict from image file
4. Exit
""")
        choice = input("Enter choice: ").strip()
        if choice == "1":
            # You can set a small limit_per_class for quick debugging
            debug_limit = None
            debug_ans = input("Train full dataset? (y/n — choose 'n' to debug with small subset): ").strip().lower()
            if debug_ans == "n":
                try:
                    limit = int(input("Enter limit images per class (e.g. 200): ").strip())
                    debug_limit = limit
                except Exception:
                    debug_limit = 200
            train_random_forest(limit_per_class=debug_limit)

        elif choice == "2":
            confirm = input("Open webcam to capture photo? (y/n): ").strip().lower()
            if confirm != "y":
                continue
            img_file = capture_from_webcam()
            if img_file:
                pred = predict_image(img_file)
                print("\nPrediction:", pred)
                try:
                    os.remove(img_file)
                except Exception:
                    pass

        elif choice == "3":
            path = input("Enter path to image file: ").strip()
            if not os.path.exists(path):
                print("[ERROR] File not found:", path)
                continue
            pred = predict_image(path)
            print("\nPrediction:", pred)

        elif choice == "4":
            print("Exiting. Goodbye!")
            break

        else:
            print("Invalid choice — try again.")

if __name__ == "__main__":
    print("Starting Skin Disease Detector (RandomForest) — Python 3.12 recommended")
    menu_loop()
