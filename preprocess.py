import os
import numpy as np
from PIL import Image

# Original dataset location
DATASET_PATH = r"C:\Users\Bhoomika\Desktop\Blood Group Detection Using Fingerprint\dataset\dataset_blood_group"

# Where processed data will be saved
OUTPUT_PATH = r"C:\Users\Bhoomika\Desktop\Blood Group Detection Using Fingerprint\dataset\processed"

# Blood group classes
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

# Fixed image size for CNN
IMAGE_SIZE = (128, 128)

# Create output folder
os.makedirs(OUTPUT_PATH, exist_ok=True)

print("===== PREPROCESSING STARTED =====")

total_images = 0

for group in blood_groups:

    input_folder = os.path.join(DATASET_PATH, group)
    output_folder = os.path.join(OUTPUT_PATH, group)

    os.makedirs(output_folder, exist_ok=True)

    print(f"\nProcessing {group}...")

    files = [
        file for file in os.listdir(input_folder)
        if file.lower().endswith((".bmp", ".jpg", ".jpeg", ".png"))
    ]

    for index, file in enumerate(files):

        input_path = os.path.join(input_folder, file)

        try:
            # Open image
            image = Image.open(input_path)

            # Convert to grayscale
            image = image.convert("L")

            # Resize to 128 x 128
            image = image.resize(IMAGE_SIZE)

            # Convert to NumPy array
            image_array = np.array(image)

            # Normalize pixels from 0-255 to 0-1
            image_array = image_array / 255.0

            # Save as NumPy file
            output_filename = os.path.splitext(file)[0] + ".npy"
            output_path = os.path.join(output_folder, output_filename)

            np.save(output_path, image_array)

            total_images += 1

            if (index + 1) % 100 == 0:
                print(f"  Processed: {index + 1}")

        except Exception as e:
            print(f"Error processing {file}: {e}")

print("\n================================")
print("PREPROCESSING COMPLETE")
print("================================")
print(f"Total images processed: {total_images}")
print("Image size: 128 x 128")
print("Image type: Grayscale")
print("Pixel range: 0 to 1")
print(f"Output folder: {OUTPUT_PATH}")
print("================================")