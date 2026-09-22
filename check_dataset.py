import os

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

total_images = 0

print("\n===== BLOOD GROUP DATASET CHECK =====\n")

for group in blood_groups:

    folder_path = os.path.join(DATASET_PATH, group)

    if not os.path.exists(folder_path):
        print(f"{group}: FOLDER NOT FOUND")
        continue

    images = [
        file for file in os.listdir(folder_path)
        if file.lower().endswith(valid_extensions)
    ]

    count = len(images)
    total_images += count

    print(f"{group:4} : {count} images")

print("\n------------------------------------")
print(f"Total images: {total_images}")
print("------------------------------------")