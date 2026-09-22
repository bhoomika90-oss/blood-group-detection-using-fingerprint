from flask import Flask, render_template, request
import os
import numpy as np
import tensorflow as tf
from PIL import Image

# =========================================================
# FLASK APP
# =========================================================

app = Flask(__name__)

# =========================================================
# PATHS
# =========================================================

MODEL_PATH = r"C:\Users\Bhoomika\Desktop\Blood Group Detection Using Fingerprint\model\blood_group_cnn.h5"

UPLOAD_FOLDER = "static/uploads"

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# =========================================================
# BLOOD GROUP CLASSES
# =========================================================

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


# =========================================================
# LOAD CNN MODEL
# =========================================================

print()
print("Loading CNN model...")

model = tf.keras.models.load_model(MODEL_PATH)

print("CNN model loaded successfully.")


# =========================================================
# PREDICTION FUNCTION
# =========================================================

def predict_image(image):

    # -----------------------------------------------------
    # Convert image to grayscale
    # -----------------------------------------------------

    image = image.convert("L")

    # -----------------------------------------------------
    # Resize to model input size
    # -----------------------------------------------------

    image = image.resize((128, 128))

    # -----------------------------------------------------
    # Convert to NumPy array
    # -----------------------------------------------------

    image_array = np.array(
        image,
        dtype=np.float32
    )

    # -----------------------------------------------------
    # Normalize pixels
    # -----------------------------------------------------

    image_array = image_array / 255.0

    # -----------------------------------------------------
    # Add CNN dimensions
    # 128 x 128
    # becomes
    # 1 x 128 x 128 x 1
    # -----------------------------------------------------

    image_array = image_array.reshape(
        1,
        128,
        128,
        1
    )

    # -----------------------------------------------------
    # CNN PREDICTION
    # -----------------------------------------------------

    prediction = model.predict(
        image_array,
        verbose=0
    )[0]

    # -----------------------------------------------------
    # Find highest probability
    # -----------------------------------------------------

    predicted_index = int(
        np.argmax(prediction)
    )

    predicted_group = blood_groups[
        predicted_index
    ]

    confidence = float(
        prediction[predicted_index] * 100
    )

    # -----------------------------------------------------
    # All probabilities
    # -----------------------------------------------------

    probabilities = []

    for i, group in enumerate(blood_groups):

        probabilities.append({

            "group": group,

            "probability": round(
                float(prediction[i] * 100),
                2
            )

        })

    return (
        predicted_group,
        round(confidence, 2),
        probabilities
    )


# =========================================================
# HOME PAGE
# =========================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# =========================================================
# CAMERA PREDICTION
# =========================================================

@app.route(
    "/predict_camera",
    methods=["POST"]
)
def predict_camera():

    try:

        # -------------------------------------------------
        # Get camera image
        # -------------------------------------------------

        file = request.files.get(
            "camera_file"
        )

        if not file:

            return render_template(
                "index.html",
                error="No camera image was received."
            )

        # -------------------------------------------------
        # Open image
        # -------------------------------------------------

        image = Image.open(file)

        print()
        print("Camera image received")
        print(
            "Original size:",
            image.size
        )
        print(
            "Image mode:",
            image.mode
        )

        # -------------------------------------------------
        # Predict
        # -------------------------------------------------

        (
            predicted_group,
            confidence,
            probabilities
        ) = predict_image(image)

        # -------------------------------------------------
        # Save camera image
        # -------------------------------------------------

        save_path = os.path.join(
            UPLOAD_FOLDER,
            "camera_fingerprint.jpg"
        )

        image.convert("RGB").save(
            save_path,
            "JPEG",
            quality=85
        )

        print(
            "Prediction:",
            predicted_group
        )

        print(
            "Confidence:",
            confidence,
            "%"
        )

        # -------------------------------------------------
        # Show result
        # -------------------------------------------------

        return render_template(

            "index.html",

            prediction=predicted_group,

            confidence=confidence,

            probabilities=probabilities,

            image_path="/" +
            save_path.replace(
                "\\",
                "/"
            ),

            source="Camera Capture"

        )

    except Exception as e:

        print()
        print(
            "Camera Error:",
            e
        )

        return render_template(

            "index.html",

            error=
            f"Camera prediction error: {e}"

        )


# =========================================================
# FILE UPLOAD PREDICTION
# =========================================================

@app.route(
    "/predict_file",
    methods=["POST"]
)
def predict_file():

    try:

        # -------------------------------------------------
        # Get uploaded file
        # -------------------------------------------------

        file = request.files.get(
            "fingerprint_file"
        )

        if not file:

            return render_template(
                "index.html",
                error="Please choose a fingerprint image."
            )

        if file.filename == "":

            return render_template(
                "index.html",
                error="No file selected."
            )

        # -------------------------------------------------
        # Open image
        # -------------------------------------------------

        image = Image.open(file)

        print()
        print("File image received")

        print(
            "File name:",
            file.filename
        )

        print(
            "Original size:",
            image.size
        )

        print(
            "Image mode:",
            image.mode
        )

        # -------------------------------------------------
        # Predict
        # -------------------------------------------------

        (
            predicted_group,
            confidence,
            probabilities
        ) = predict_image(image)

        # -------------------------------------------------
        # Save uploaded image
        # -------------------------------------------------

        save_path = os.path.join(
            UPLOAD_FOLDER,
            "uploaded_fingerprint.jpg"
        )

        image.convert("RGB").save(
            save_path,
            "JPEG",
            quality=85
        )

        print(
            "Prediction:",
            predicted_group
        )

        print(
            "Confidence:",
            confidence,
            "%"
        )

        # -------------------------------------------------
        # Show result
        # -------------------------------------------------

        return render_template(

            "index.html",

            prediction=predicted_group,

            confidence=confidence,

            probabilities=probabilities,

            image_path="/" +
            save_path.replace(
                "\\",
                "/"
            ),

            source="File Upload"

        )

    except Exception as e:

        print()
        print(
            "File Error:",
            e
        )

        return render_template(

            "index.html",

            error=
            f"File prediction error: {e}"

        )


# =========================================================
# RUN FLASK
# =========================================================

if __name__ == "__main__":

    print()
    print(
        "=========================================="
    )

    print(
        " BLOOD GROUP DETECTION USING FINGERPRINT"
    )

    print(
        "=========================================="
    )

    print(
        "Camera + File Upload"
    )

    print(
        "=========================================="
    )

    print(
        "Open in browser:"
    )

    print(
        "http://127.0.0.1:5000"
    )

    print(
        "=========================================="
    )

    app.run(

        host="127.0.0.1",

        port=5000,

        debug=True

    )