import os
import shutil
import random

SOURCE_PATH = r"C:\Users\Bhoomika\Desktop\Blood Group Detection Using Fingerprint\dataset\processed"

OUTPUT_PATH = r"C:\Users\Bhoomika\Desktop\Blood Group Detection Using Fingerprint\dataset\split"

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

TRAIN_RATIO = 0.80

random.seed(42)

print("===== DATASET SPLITTING STARTED =====")

total_train = 0
total_test = 0

for group in blood_groups:

    source_folder = os.path.join(SOURCE_PATH, group)

    train_folder = os.path.join(OUTPUT_PATH, "train", group)
    test_folder = os.path.join(OUTPUT_PATH, "test", group)

    os.makedirs(train_folder, exist_ok=True)
    os.makedirs(test_folder, exist_ok=True)

    files = [
        file for file in os.listdir(source_folder)
        if file.endswith(".npy")
    ]

    random.shuffle(files)

    train_count = int(len(files) * TRAIN_RATIO)

    train_files = files[:train_count]
    test_files = files[train_count:]

    print(f"\n{group}")
    print(f"Total : {len(files)}")
    print(f"Train : {len(train_files)}")
    print(f"Test  : {len(test_files)}")

    for file in train_files:
        source = os.path.join(source_folder, file)
        destination = os.path.join(train_folder, file)
        shutil.copy2(source, destination)

    for file in test_files:
        source = os.path.join(source_folder, file)
        destination = os.path.join(test_folder, file)
        shutil.copy2(source, destination)

    total_train += len(train_files)
    total_test += len(test_files)

print("\n====================================")
print("DATASET SPLITTING COMPLETE")
print("====================================")
print(f"Total training images: {total_train}")
print(f"Total testing images : {total_test}")
print(f"Total images         : {total_train + total_test}")
print("====================================")