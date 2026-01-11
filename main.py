import os
import cv2
import numpy as np
import time
import opendatasets as od 
from skimage.feature import hog
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import accuracy_score, classification_report

# Configuration
DATASET_URL = "https://www.kaggle.com/datasets/jiayuanchengala/aid-scene-classification-datasets"
DATA_ROOT = "aid-scene-classification-datasets"
DATA_DIR = os.path.join(DATA_ROOT, "AID")

IMG_SIZE = 64
TEST_SPLIT = 0.2
SEED = 42
KERNEL = 'rbf'
PIXELS_PER_CELL = (8, 8)
OUTPUT_FILE = "results.txt"

# Download the dataset (If needed)
if not os.path.exists(DATA_ROOT):
    od.download(DATASET_URL)

def extract_color_histogram(image, bins=32):
    # Calculate histogram for each channel (R, G, B)
    hist_r = cv2.calcHist([image], [0], None, [bins], [0, 256])
    hist_g = cv2.calcHist([image], [1], None, [bins], [0, 256])
    hist_b = cv2.calcHist([image], [2], None, [bins], [0, 256])

    # Normalize and flatten
    cv2.normalize(hist_r, hist_r)
    cv2.normalize(hist_g, hist_g)
    cv2.normalize(hist_b, hist_b)

    return np.concatenate([hist_r, hist_g, hist_b]).flatten()

data_features = []
labels = []

classes = os.listdir(DATA_DIR)
start_time = time.time()

for category in classes:
    path = os.path.join(DATA_DIR, category)
    if not os.path.isdir(path): continue

    print(f"Processing {category}...")
    for img_name in os.listdir(path):
        img_path = os.path.join(path, img_name)
        img = cv2.imread(img_path)
        if img is None: continue

        # Resize the image
        img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

        # Extract HOG 
        hog_feat = hog(img, orientations=9, pixels_per_cell=PIXELS_PER_CELL,
                       cells_per_block=(2, 2), channel_axis=-1)

        # Extract Color Histogram 
        color_feat = extract_color_histogram(img)

        # Concatenate them
        combined_feat = np.hstack([hog_feat, color_feat])

        data_features.append(combined_feat)
        labels.append(category)

print(f"Feature Extraction Done. Time: {time.time() - start_time:.2f}s")

# Prepare Data
X = np.array(data_features)
y = np.array(labels)

# Label Encoding
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Split
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=TEST_SPLIT, random_state=SEED)

# Scale Data
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Train SVM
training_start_time = time.time()
svm = SVC(kernel=KERNEL, C=10, gamma='scale', random_state=SEED)
svm.fit(X_train, y_train)
training_time = time.time() - training_start_time

# Evaluation
y_pred = svm.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
report = classification_report(y_test, y_pred, target_names=le.classes_)

# Print the result
print(f"\nACCURACY: {accuracy * 100:.2f}%")
print(f"TRAINING TIME: {training_time:.2f} seconds")
print(report)

# Save Results
with open(OUTPUT_FILE, "w") as f:
    config_line = (f"CONFIG: IMG_SIZE={IMG_SIZE}, TEST_SPLIT={TEST_SPLIT}, "
                   f"SEED={SEED}, KERNEL={KERNEL}, PIXELS_PER_CELL={PIXELS_PER_CELL}\n")
    f.write(config_line)
    f.write(f"\nACCURACY: {accuracy * 100:.2f}% | TRAINING TIME: {training_time:.2f}s\n")
    f.write("\n" + report)

print(f"\nResults saved to {OUTPUT_FILE}")