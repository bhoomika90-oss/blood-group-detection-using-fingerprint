import os
import matplotlib.pyplot as plt
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

plt.figure(figsize=(12, 8))

plot_number = 1

for group in blood_groups:

    folder_path = os.path.join(DATASET_PATH, group)

    files = [
        file for file in os.listdir(folder_path)
        if file.lower().endswith((".bmp", ".jpg", ".jpeg", ".png"))
    ]

    image_path = os.path.join(folder_path, files[0])

    image = Image.open(image_path)

    plt.subplot(2, 4, plot_number)
    plt.imshow(image)
    plt.title(group)
    plt.axis("off")

    plot_number += 1

plt.tight_layout()
plt.show()