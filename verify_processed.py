import os
import numpy as np

PROCESSED_PATH = r"C:\Users\Bhoomika\Desktop\Blood Group Detection Using Fingerprint\dataset\processed"

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

print("===== PROCESSED DATA VERIFICATION =====")

total = 0

for group in blood_groups:

    folder = os.path.join(PROCESSED_PATH, group)

    files = [
        file for file in os.listdir(folder)
        if file.endswith(".npy")
    ]

    print(f"{group:4} : {len(files)} files")

    # Check first file
    if len(files) > 0:

        sample_path = os.path.join(folder, files[0])

        data = np.load(sample_path)

        print(f"      Shape: {data.shape}")
        print(f"      Minimum: {data.min():.4f}")
        print(f"      Maximum: {data.max():.4f}")

        total += len(files)

print("----------------------------------------")
print(f"Total processed files: {total}")
print("========================================")