import os
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from sklearn.utils.class_weight import compute_class_weight

# =====================================================
# PATHS
# =====================================================

TRAIN_PATH = r"C:\Users\Bhoomika\Desktop\Blood Group Detection Using Fingerprint\dataset\split\train"
TEST_PATH = r"C:\Users\Bhoomika\Desktop\Blood Group Detection Using Fingerprint\dataset\split\test"

MODEL_PATH = r"C:\Users\Bhoomika\Desktop\Blood Group Detection Using Fingerprint\model"

os.makedirs(MODEL_PATH, exist_ok=True)

# =====================================================
# BLOOD GROUP CLASSES
# =====================================================

blood_groups = [
    "A-",
    "A+",
    "AB-",
    "AB+",
    "B-",
    "B+",
    "O-",
    "O+"
]

class_to_index = {
    group: index
    for index, group in enumerate(blood_groups)
}

print("==========================================")
print("      BLOOD GROUP CNN TRAINING")
print("==========================================")

print("\nClasses:")

for group, index in class_to_index.items():
    print(f"{index} -> {group}")

# =====================================================
# LOAD DATA
# =====================================================

def load_dataset(folder_path):

    images = []
    labels = []

    for group in blood_groups:

        group_folder = os.path.join(folder_path, group)

        files = [
            file for file in os.listdir(group_folder)
            if file.endswith(".npy")
        ]

        print(f"Loading {group}: {len(files)} images")

        for file in files:

            file_path = os.path.join(group_folder, file)

            image = np.load(file_path)

            # Add channel dimension
            image = image.reshape(128, 128, 1)

            images.append(image)
            labels.append(class_to_index[group])

    return np.array(images, dtype=np.float32), np.array(labels)


print("\nLoading training dataset...")

X_train, y_train = load_dataset(TRAIN_PATH)

print("\nLoading testing dataset...")

X_test, y_test = load_dataset(TEST_PATH)

print("\nDataset loaded successfully.")

print("Training shape:", X_train.shape)
print("Testing shape :", X_test.shape)

# =====================================================
# CLASS WEIGHTS
# =====================================================

class_weights_array = compute_class_weight(
    class_weight="balanced",
    classes=np.arange(len(blood_groups)),
    y=y_train
)

class_weights = {
    index: weight
    for index, weight in enumerate(class_weights_array)
}

print("\nClass weights:")

for index, weight in class_weights.items():
    print(f"{blood_groups[index]} : {weight:.4f}")

# =====================================================
# CNN MODEL
# =====================================================

model = models.Sequential([

    layers.Input(shape=(128, 128, 1)),

    # First convolution block
    layers.Conv2D(
        32,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D((2, 2)),

    # Second convolution block
    layers.Conv2D(
        64,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D((2, 2)),

    # Third convolution block
    layers.Conv2D(
        128,
        (3, 3),
        activation="relu",
        padding="same"
    ),

    layers.MaxPooling2D((2, 2)),

    # Convert feature maps into vector
    layers.Flatten(),

    # Fully connected layer
    layers.Dense(128, activation="relu"),

    # Dropout helps reduce overfitting
    layers.Dropout(0.5),

    # 8 output classes
    layers.Dense(
        8,
        activation="softmax"
    )
])

# =====================================================
# COMPILE MODEL
# =====================================================

model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

print("\n==========================================")
print("MODEL ARCHITECTURE")
print("==========================================")

model.summary()

# =====================================================
# TRAIN MODEL
# =====================================================

print("\n==========================================")
print("TRAINING STARTED")
print("==========================================")

history = model.fit(
    X_train,
    y_train,
    validation_data=(X_test, y_test),
    epochs=15,
    batch_size=32,
    class_weight=class_weights,
    verbose=1
)

# =====================================================
# SAVE MODEL
# =====================================================

model_file = os.path.join(
    MODEL_PATH,
    "blood_group_cnn.h5"
)

model.save(model_file)

print("\n==========================================")
print("TRAINING COMPLETE")
print("==========================================")

print(f"Model saved to:")
print(model_file)