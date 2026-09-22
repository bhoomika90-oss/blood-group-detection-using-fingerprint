import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    ConfusionMatrixDisplay
)

# =====================================================
# PATHS
# =====================================================

TEST_PATH = r"C:\Users\Bhoomika\Desktop\Blood Group Detection Using Fingerprint\dataset\split\test"

MODEL_PATH = r"C:\Users\Bhoomika\Desktop\Blood Group Detection Using Fingerprint\model\blood_group_cnn.h5"

RESULT_PATH = r"C:\Users\Bhoomika\Desktop\Blood Group Detection Using Fingerprint\model"

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

class_to_index = {
    group: index
    for index, group in enumerate(blood_groups)
}

# =====================================================
# LOAD TEST DATA
# =====================================================

print("==========================================")
print("       CNN MODEL EVALUATION")
print("==========================================")

images = []
labels = []

print("\nLoading test images...")

for group in blood_groups:

    folder = os.path.join(TEST_PATH, group)

    files = [
        file for file in os.listdir(folder)
        if file.endswith(".npy")
    ]

    print(f"{group}: {len(files)} images")

    for file in files:

        file_path = os.path.join(folder, file)

        image = np.load(file_path)

        # Add channel dimension
        image = image.reshape(128, 128, 1)

        images.append(image)
        labels.append(class_to_index[group])

X_test = np.array(images, dtype=np.float32)
y_test = np.array(labels)

print("\nTest data shape:", X_test.shape)
print("Test labels shape:", y_test.shape)

# =====================================================
# LOAD TRAINED MODEL
# =====================================================

print("\nLoading trained CNN model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("Model loaded successfully.")

# =====================================================
# TEST LOSS AND ACCURACY
# =====================================================

print("\n==========================================")
print("TESTING MODEL")
print("==========================================")

test_loss, test_accuracy = model.evaluate(
    X_test,
    y_test,
    verbose=1
)

print("\n==========================================")
print("FINAL TEST RESULTS")
print("==========================================")

print(f"Test Loss     : {test_loss:.4f}")
print(f"Test Accuracy : {test_accuracy * 100:.2f}%")

# =====================================================
# PREDICTIONS
# =====================================================

print("\nGenerating predictions...")

predictions = model.predict(
    X_test,
    verbose=1
)

predicted_labels = np.argmax(
    predictions,
    axis=1
)

# =====================================================
# CLASSIFICATION REPORT
# =====================================================

print("\n==========================================")
print("CLASSIFICATION REPORT")
print("==========================================")

report = classification_report(
    y_test,
    predicted_labels,
    target_names=blood_groups,
    digits=4
)

print(report)

# Save report
report_path = os.path.join(
    RESULT_PATH,
    "classification_report.txt"
)

with open(report_path, "w") as file:
    file.write("BLOOD GROUP CNN CLASSIFICATION REPORT\n")
    file.write("=====================================\n\n")
    file.write(f"Test Loss: {test_loss:.4f}\n")
    file.write(f"Test Accuracy: {test_accuracy * 100:.2f}%\n\n")
    file.write(report)

print(f"Report saved to:\n{report_path}")

# =====================================================
# CONFUSION MATRIX
# =====================================================

print("\nCreating confusion matrix...")

cm = confusion_matrix(
    y_test,
    predicted_labels
)

print("\nConfusion Matrix:")
print(cm)

plt.figure(figsize=(10, 8))

display = ConfusionMatrixDisplay(
    confusion_matrix=cm,
    display_labels=blood_groups
)

display.plot(
    xticks_rotation=45
)

plt.title("Blood Group CNN - Confusion Matrix")
plt.tight_layout()

confusion_path = os.path.join(
    RESULT_PATH,
    "confusion_matrix.png"
)

plt.savefig(confusion_path, dpi=300)

plt.show()

print("\nConfusion matrix saved to:")
print(confusion_path)

print("\n==========================================")
print("EVALUATION COMPLETE")
print("==========================================")