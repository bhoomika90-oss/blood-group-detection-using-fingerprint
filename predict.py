import os
import numpy as np
import tensorflow as tf
from PIL import Image

# =====================================================
# MODEL PATH
# =====================================================

MODEL_PATH = r"C:\Users\Bhoomika\Desktop\Blood Group Detection Using Fingerprint\model\blood_group_cnn.h5"

# =====================================================
# CLASSES
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

# =====================================================
# LOAD MODEL
# =====================================================

print("Loading CNN model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# =====================================================
# ASK FOR IMAGE
# =====================================================

image_path = input(
    "\nEnter the full path of a fingerprint image: "
).strip().strip('"')

# =====================================================
# CHECK IMAGE
# =====================================================

if not os.path.exists(image_path):

    print("\nERROR: Image file not found.")

    exit()

# =====================================================
# PREPROCESS IMAGE
# =====================================================

image = Image.open(image_path)

print("\nOriginal image information:")
print("Size :", image.size)
print("Mode :", image.mode)

# Convert to grayscale
image = image.convert("L")

# Resize
image = image.resize((128, 128))

# Convert to NumPy
image_array = np.array(image, dtype=np.float32)

# Normalize
image_array = image_array / 255.0

# Add dimensions:
# 128 x 128
#       ↓
# 128 x 128 x 1
#       ↓
# 1 x 128 x 128 x 1

image_array = image_array.reshape(1, 128, 128, 1)

# =====================================================
# PREDICT
# =====================================================

print("\nAnalyzing fingerprint...")

prediction = model.predict(
    image_array,
    verbose=0
)

predicted_index = np.argmax(prediction[0])

predicted_group = blood_groups[predicted_index]

confidence = prediction[0][predicted_index] * 100

# =====================================================
# RESULT
# =====================================================

print("\n==========================================")
print("          PREDICTION RESULT")
print("==========================================")

print(f"Predicted label : {predicted_group}")
print(f"Model confidence: {confidence:.2f}%")

print("==========================================")

print("\nAll class probabilities:")

for i, group in enumerate(blood_groups):

    probability = prediction[0][i] * 100

    print(
        f"{group:4} : {probability:.2f}%"
    )

print("\nNOTE:")
print("This is an experimental ML prediction based")
print("on the labeled dataset and is NOT a clinical")
print("blood-group test.")