import os
import cv2
import pickle
import numpy as np
import matplotlib.pyplot as plt

from skimage.feature import local_binary_pattern, hog, graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, ConfusionMatrixDisplay

# ---------------- CONFIG -----------------
DATA_PATH = "SKID"
MODEL_PATH = "svm_best.pkl"
SCALER_PATH = "scaler.pkl"
CLASSES_PATH = "classes.txt"

IMAGE_SIZE = (200, 200)
LBP_POINTS = 24
LBP_RADIUS = 3
VALID_EXT = (".jpg", ".jpeg", ".png")

# ---------------- FEATURE EXTRACTION -----------------

def preprocess_image(image):
    """Apply smoothing + normalization to improve feature consistency."""
    image = cv2.GaussianBlur(image, (5, 5), 0)
    image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX)
    return image

def extract_lbp(gray):
    lbp = local_binary_pattern(gray, LBP_POINTS, LBP_RADIUS, method="uniform")
    (hist, _) = np.histogram(lbp.ravel(), bins=256, range=(0, 256))
    hist = hist.astype("float")
    hist /= (hist.sum() + 1e-6)
    return hist

def extract_hog(gray):
    hog_features = hog(gray, orientations=9, pixels_per_cell=(32, 32),
                       cells_per_block=(1, 1), block_norm="L2-Hys")
    return hog_features

def extract_glcm(gray):
    gl = graycomatrix(gray, [5], [0], levels=256, symmetric=True, normed=True)
    props = ["contrast", "dissimilarity", "homogeneity", "energy", "correlation"]
    return [graycoprops(gl, p)[0, 0] for p in props]

def extract_features(path):
    try:
        img = cv2.imread(path)
        if img is None:
            print("Could not read:", path)
            return None

        img = cv2.resize(img, IMAGE_SIZE)
        img = preprocess_image(img)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        f_hog = extract_hog(gray)
        f_lbp = extract_lbp(gray)
        f_glcm = extract_glcm(gray)

        return np.hstack([f_hog, f_lbp, f_glcm])

    except Exception as e:
        print("Feature error:", e)
        return None

# ---------------- LOAD DATA -----------------

def load_data():
    X, y, class_names = [], [], []

    if not os.path.exists(DATA_PATH):
        print("Dataset not found:", DATA_PATH)
        return [], [], []

    classes = sorted(os.listdir(DATA_PATH))
    
    for label, cname in enumerate(classes):
        cpath = os.path.join(DATA_PATH, cname)
        if not os.path.isdir(cpath): continue

        print("\nProcessing class:", cname)
        class_names.append(cname)

        for img_name in os.listdir(cpath):
            if not img_name.lower().endswith(VALID_EXT): continue

            path = os.path.join(cpath, img_name)
            feat = extract_features(path)

            if feat is not None:
                X.append(feat)
                y.append(label)

    return np.array(X), np.array(y), class_names

# ---------------- TRAIN MODEL -----------------

def train_model():
    X, y, class_names = load_data()
    print("\nTotal Samples:", len(X))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    # Hyperparameter tuning
    print("\n Tuning SVM parameters...")
    params = {"C": [0.1, 1, 10], "gamma": ["scale", "auto"], "kernel": ["rbf"]}
    
    grid = GridSearchCV(SVC(class_weight="balanced"), params, cv=3, n_jobs=-1)
    grid.fit(X_train, y_train)

    model = grid.best_estimator_
    print("\nBest SVM:", model)

    y_pred = model.predict(X_test)
    acc = accuracy_score(y_test, y_pred)

    print("\nAccuracy:", acc * 100)
    print("\nClassification Report:\n")
    print(classification_report(y_test, y_pred, target_names=class_names))

    ConfusionMatrixDisplay.from_predictions(y_test, y_pred)
    plt.show()

    pickle.dump(model, open(MODEL_PATH, "wb"))
    pickle.dump(scaler, open(SCALER_PATH, "wb"))
    open(CLASSES_PATH, "w").write("\n".join(class_names))

    print("\nModel saved!")

# ---------------- PREDICT -----------------

def predict_image(path):
    model = pickle.load(open(MODEL_PATH, "rb"))
    scaler = pickle.load(open(SCALER_PATH, "rb"))
    classes = open(CLASSES_PATH).read().splitlines()

    feat = extract_features(path)
    if feat is None:
        return "Error processing image."

    feat = scaler.transform([feat])
    pred = model.predict(feat)[0]
    return classes[pred]

# ---------------- WEBCAM -----------------

def capture_image():
    cam = cv2.VideoCapture(0)
    print("\nPress 's' to take photo, 'q' to cancel.")

    while True:
        ret, frame = cam.read()
        cv2.imshow("Webcam", frame)
        key = cv2.waitKey(1) & 0xFF
        
        if key == ord('s'):
            cv2.imwrite("capture.jpg", frame)
            cam.release()
            cv2.destroyAllWindows()
            return "capture.jpg"

        elif key == ord('q'):
            cam.release()
            cv2.destroyAllWindows()
            return None

# ---------------- MENU LOOP -----------------

def menu():
    while True:
        print("""
===============================
  SKIN DISEASE DETECTION (ML)
===============================
1. Train Model
2. Predict from Webcam
3. Predict from Image File
4. Exit
""")

        choice = input("Enter choice: ").strip()

        if choice == "1":
            train_model()

        elif choice == "2":
            img = capture_image()
            if img:
                print("\nPrediction:", predict_image(img))

        elif choice == "3":
            path = input("Enter image path: ")
            print("\nPrediction:", predict_image(path))

        elif choice == "4":
            print("Goodbye!")
            break

        else:
            print("Invalid choice, try again.")

# START PROGRAM
menu()
