import tensorflow as tf
from tensorflow.keras import layers, models
import os
import cv2
import numpy as np
from sklearn.model_selection import train_test_split

# Constants
IMG_SIZE = 128
DATASET_PATH = r"C:\Users\yuvra\Downloads\brain tumor\brain tumor dataset\Training"
CATEGORIES = ["glioma_tumor", "meningioma_tumor", "no_tumor", "pituitary_tumor"]

def load_images_from_folder(folder):
    images, labels = [], []

    if not os.path.exists(folder):
        raise FileNotFoundError(f"Dataset path not found: {folder}")

    for category in CATEGORIES:
        category_path = os.path.join(folder, category)

        if not os.path.exists(category_path):
            raise FileNotFoundError(f"Category folder not found: {category_path}")

        class_num = CATEGORIES.index(category)

        for img_name in os.listdir(category_path):
            try:
                img_path = os.path.join(category_path, img_name)
                img = cv2.imread(img_path)

                if img is None:
                    print(f"Skipping unreadable image: {img_path}")
                    continue

                img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                img = img.astype(np.float32) / 255.0

                images.append(img)
                labels.append(class_num)

            except Exception as e:
                print(f"Error loading image {img_name}: {e}")

    return np.array(images, dtype=np.float32), np.array(labels)

print("Loading dataset...")
X, y = load_images_from_folder(DATASET_PATH)

print("Total images loaded:", len(X))

X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.3, random_state=42, stratify=y
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

model = models.Sequential([
    layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
    layers.Conv2D(32, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(len(CATEGORIES), activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

history = model.fit(
    X_train, y_train,
    epochs=10,
    batch_size=32,
    validation_data=(X_val, y_val)
)

model.save("Brain_Tumor_Detection_Model.keras")
print("Model saved successfully.")

test_loss, test_acc = model.evaluate(X_test, y_test, verbose=2)
print(f"\nTest Accuracy: {test_acc * 100:.2f}%")