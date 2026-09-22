import os
from PIL import Image
from collections import Counter

DATASET_PATH = r"C:\Users\Bhoomika\Desktop\Blood Group Detection Using Fingerprint\dataset\dataset_blood_group"

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

valid_extensions = (".bmp", ".jpg", ".jpeg", ".png")

sizes = Counter()
modes = Counter()

print("\n===== IMAGE INFORMATION =====\n")

for group in blood_groups:

    folder_path = os.path.join(DATASET_PATH, group)

    for filename in os.listdir(folder_path):

        if not filename.lower().endswith(valid_extensions):
            continue

        image_path = os.path.join(folder_path, filename)

        with Image.open(image_path) as img:
            sizes[img.size] += 1
            modes[img.mode] += 1

print("Image sizes:")
for size, count in sizes.most_common():
    print(f"{size} : {count} images")

print("\nImage modes:")
for mode, count in modes.items():
    print(f"{mode} : {count} images")

print("\n===== CHECK COMPLETE =====")