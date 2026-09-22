import os
from PIL import Image

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

print("\n===== IMAGE QUALITY CHECK =====\n")

total_checked = 0
corrupted = 0

for group in blood_groups:

    folder_path = os.path.join(DATASET_PATH, group)

    print(f"Checking {group}...")

    for filename in os.listdir(folder_path):

        if not filename.lower().endswith(valid_extensions):
            continue

        image_path = os.path.join(folder_path, filename)

        try:
            with Image.open(image_path) as img:
                img.verify()

            total_checked += 1

        except Exception:
            corrupted += 1
            print("Corrupted:", image_path)

print("\n--------------------------------")
print("Images checked :", total_checked)
print("Corrupted      :", corrupted)
print("--------------------------------")