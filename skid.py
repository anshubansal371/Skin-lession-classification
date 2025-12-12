import os
import cv2
import pickle
import numpy as np
from skimage.feature import graycomatrix, graycoprops
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
#to run the program run this in terminal
#py -3.12 a:\python\skid.py
# --- Main Configuration ---
DATA_PATH = 'SKID'
MODEL_PATH = 'skin_disease_model_svm.pkl'
SCALER_PATH = 'scaler_svm.pkl'
CLASS_NAMES_PATH = 'class_names_svm.txt'

# --- 1. Feature Extraction ---
def extract_features(image_path):
    """Extracts color and texture features from an image."""
    try:
        image = cv2.imread(image_path)
        image = cv2.resize(image, (200, 200))
        
        # Color Features
        hist = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
        cv2.normalize(hist, hist)
        color_features = hist.flatten()
        
        # Texture Features
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        glcm = graycomatrix(gray, distances=[5], angles=[0], levels=256, symmetric=True, normed=True)
        props_to_calc = ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']
        texture_features = [graycoprops(glcm, prop)[0, 0] for prop in props_to_calc]

        return np.hstack([color_features, texture_features])
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

# --- 2. Data Loading ---
def load_data(data_path):
    """Loads all images, extracts features, and assigns labels."""
    X, y, class_names = [], [], []
    print("Loading images and extracting features. This may take a while...")
    class_dirs = sorted([d for d in os.listdir(data_path) if os.path.isdir(os.path.join(data_path, d))])
    
    for class_idx, class_name in enumerate(class_dirs):
        class_dir = os.path.join(data_path, class_name)
        class_names.append(class_name)
        print(f"Processing class ({class_idx+1}/{len(class_dirs)}): {class_name}")
        for file_name in os.listdir(class_dir):
            file_path = os.path.join(class_dir, file_name)
            features = extract_features(file_path)
            if features is not None:
                X.append(features)
                y.append(class_idx)
    
    print(f"Finished loading. Found {len(X)} images in {len(class_names)} classes.")
    return np.array(X), np.array(y), class_names

# --- 3. Live Capture and Prediction ---
def capture_from_webcam(filename="live_capture.jpg"):
    """Captures a single frame from the webcam and saves it."""
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return None

    print("\nWebcam opened. Press 's' to save and quit, or 'q' to quit without saving.")
    while True:
        ret, frame = cap.read()
        if not ret: break
        cv2.imshow('Webcam - Press "s" to save, "q" to quit', frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord('s'):
            cv2.imwrite(filename, frame)
            print(f"Image saved as {filename}")
            break
        elif key == ord('q'):
            filename = None
            print("Quitting without saving.")
            break
            
    cap.release()
    cv2.destroyAllWindows()
    return filename

def predict_disease(file_path, model, scaler, class_names):
    """Predicts the disease for a single image file."""
    features = extract_features(file_path).reshape(1, -1)
    features_scaled = scaler.transform(features)
    prediction_idx = model.predict(features_scaled)[0]
    return class_names[prediction_idx]

# --- Main Script Logic ---
if not os.path.exists(MODEL_PATH):
    print("No trained model found. Starting training process...")
    X, y, CLASS_NAMES = load_data(DATA_PATH)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    print("\nTraining the SVM model...")
    model = SVC(kernel='linear', C=1.0, random_state=42) 
    model.fit(X_train_scaled, y_train)
    print("Training finished.")
    
    y_pred = model.predict(X_test_scaled)
    accuracy = accuracy_score(y_test, y_pred)
    print(f"\nModel Accuracy on Test Set: {accuracy * 100:.2f}%\n")
    
    with open(MODEL_PATH, 'wb') as f: pickle.dump(model, f)
    with open(SCALER_PATH, 'wb') as f: pickle.dump(scaler, f)
    with open(CLASS_NAMES_PATH, 'w') as f: f.write('\n'.join(CLASS_NAMES))
    print(f"Model, scaler, and class names saved successfully.")
else:
    print(f"Loading pre-trained model and scaler...")
    with open(MODEL_PATH, 'rb') as f: model = pickle.load(f)
    with open(SCALER_PATH, 'rb') as f: scaler = pickle.load(f)
    with open(CLASS_NAMES_PATH, 'r') as f: CLASS_NAMES = [line.strip() for line in f]
    print(f"Model and classes '{CLASS_NAMES}' loaded successfully.")

# --- Prediction Section ---
print("\n--- Ready for Prediction ---")
input("Press Enter to capture an image from your webcam...")
live_image_file = capture_from_webcam()
if live_image_file:
    predicted_disease = predict_disease(live_image_file, model, scaler, CLASS_NAMES)
    print(f"\nPrediction Result:\n-> Detected Disease: {predicted_disease}")
    os.remove(live_image_file)