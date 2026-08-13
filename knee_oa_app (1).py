"""
==============================================================
 Knee Osteoarthritis Detection using Deep Learning
 Single-file version - everything in one Python script
==============================================================

Tech used: TensorFlow, NumPy, Pandas, Streamlit

WHAT'S IN THIS FILE
--------------------
  1. CNN model definition          (build_model)
  2. Training function              (train_model)
  3. Prediction function            (predict_grade)
  4. Website / UI                   (Streamlit, runs at the bottom)

HOW TO RUN THE WEBSITE
------------------------
    pip install tensorflow numpy pandas pillow streamlit
    streamlit run knee_oa_app.py

HOW TO TRAIN ON YOUR OWN DATASET (run this from a python shell,
or uncomment the train_model() call at the bottom of this file)
------------------------------------------------------------------
  1. Get a knee X-ray dataset, e.g.:
     https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity

  2. Arrange it like this:
        dataset/
          images/
            img001.png
            img002.png
            ...
          labels.csv        <- columns: filename,label   (label = 0 to 4)

  3. Run:  python knee_oa_app.py --train
     This trains the CNN and saves it to model/kl_model.h5

  4. Then run:  streamlit run knee_oa_app.py
     The website will automatically use your trained model instead
     of demo mode.
==============================================================
"""

import os
import sys
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import load_img, img_to_array

# ----------------------------------------------------------------
# CONFIG
# ----------------------------------------------------------------
IMG_SIZE = (128, 128)          # image size fed into the CNN
NUM_CLASSES = 5                 # KL Grade 0, 1, 2, 3, 4
CSV_PATH = "dataset/labels.csv"
IMAGES_FOLDER = "dataset/images"
MODEL_PATH = "model/kl_model.h5"
EPOCHS = 15
BATCH_SIZE = 16

KL_LABELS = {
    0: ("Grade 0", "Normal", "No signs of knee osteoarthritis."),
    1: ("Grade 1", "Doubtful", "Very early signs; doubtful joint narrowing or small bone spurs."),
    2: ("Grade 2", "Mild", "Definite bone spurs and slight joint space narrowing."),
    3: ("Grade 3", "Moderate", "Clear joint space narrowing, multiple bone spurs, some bone changes."),
    4: ("Grade 4", "Severe", "Large bone spurs, severe joint space narrowing, major bone damage."),
}


# ----------------------------------------------------------------
# 1. MODEL DEFINITION
# ----------------------------------------------------------------
def build_model():
    """
    A simple CNN (Convolutional Neural Network):
        Conv2D + MaxPooling  -> learns simple edges/textures
        Conv2D + MaxPooling  -> learns more complex patterns
        Conv2D + MaxPooling  -> learns joint-space / bone-spur features
        Flatten -> Dense -> Dropout -> Output (5 KL grades)
    """
    model = models.Sequential([
        layers.Input(shape=(IMG_SIZE[0], IMG_SIZE[1], 3)),

        layers.Conv2D(32, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),

        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D(2, 2),

        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dropout(0.3),
        layers.Dense(NUM_CLASSES, activation="softmax"),
    ])

    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model


# ----------------------------------------------------------------
# 2. TRAINING
# ----------------------------------------------------------------
def load_dataset():
    """Reads labels.csv with pandas and loads each image + label into numpy arrays."""
    df = pd.read_csv(CSV_PATH)
    print(f"Found {len(df)} images in {CSV_PATH}")
    print(df["label"].value_counts().sort_index())

    images, labels = [], []
    for _, row in df.iterrows():
        img_path = f"{IMAGES_FOLDER}/{row['filename']}"
        img = load_img(img_path, target_size=IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        images.append(img_array)
        labels.append(row["label"])

    return np.array(images), np.array(labels)


def train_model():
    X, y = load_dataset()

    split_index = int(len(X) * 0.8)
    X_train, X_val = X[:split_index], X[split_index:]
    y_train, y_val = y[:split_index], y[split_index:]

    print(f"Training on {len(X_train)} images, validating on {len(X_val)} images")

    model = build_model()
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE
    )

    os.makedirs("model", exist_ok=True)
    model.save(MODEL_PATH)
    print(f"\nModel saved to {MODEL_PATH}")
    print("Final validation accuracy:", history.history["val_accuracy"][-1])


# ----------------------------------------------------------------
# 3. PREDICTION
# ----------------------------------------------------------------
_loaded_model = None  # cached so we don't reload the model on every call


def get_model():
    """Loads the trained model if it exists, otherwise returns None (demo mode)."""
    global _loaded_model
    if _loaded_model is not None:
        return _loaded_model
    if os.path.exists(MODEL_PATH):
        _loaded_model = tf.keras.models.load_model(MODEL_PATH)
    return _loaded_model


def predict_grade(image):
    """
    Takes a PIL image, returns (grade_label, name, description, confidence, demo_mode)
    """
    model = get_model()

    if model is None:
        # DEMO MODE - no trained model yet, return a random placeholder
        predicted_class = int(np.random.choice(list(KL_LABELS.keys())))
        confidence = round(float(np.random.uniform(0.6, 0.95)), 2)
        demo = True
    else:
        img = image.convert("RGB").resize(IMG_SIZE)
        img_array = img_to_array(img) / 255.0
        img_array = np.expand_dims(img_array, axis=0)
        predictions = model.predict(img_array)[0]
        predicted_class = int(np.argmax(predictions))
        confidence = round(float(np.max(predictions)), 2)
        demo = False

    grade, name, desc = KL_LABELS[predicted_class]
    return grade, name, desc, confidence, demo


def predict_from_path(image_path):
    """Command-line helper: predict a single image file given its path."""
    from PIL import Image
    image = Image.open(image_path)
    grade, name, desc, confidence, demo = predict_grade(image)

    print(f"\nPredicted: {grade} - {name}")
    print(f"Description: {desc}")
    print(f"Confidence: {confidence * 100:.1f}%")
    if demo:
        print("(DEMO MODE - no trained model found, this is a placeholder result)")


# ----------------------------------------------------------------
# 4. WEBSITE (Streamlit UI)
# ----------------------------------------------------------------
def run_website():
    import streamlit as st
    from PIL import Image

    st.set_page_config(page_title="Knee Osteoarthritis Detection", page_icon="🦴")

    st.title("🦴 Knee Osteoarthritis Detection")
    st.write("Upload a knee X-ray image to predict its KL (Kellgren-Lawrence) grade.")

    if get_model() is None:
        st.warning(
            "No trained model found at model/kl_model.h5 - running in DEMO MODE "
            "(predictions are random placeholders). Train the model first "
            "(see instructions at the top of this file) to enable real predictions."
        )

    uploaded_file = st.file_uploader("Upload knee X-ray", type=["png", "jpg", "jpeg"])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded X-ray", use_container_width=True)

        if st.button("Predict"):
            grade, name, desc, confidence, demo = predict_grade(image)

            st.subheader(f"{grade} — {name}")
            st.write(desc)
            st.write(f"**Confidence:** {confidence * 100:.0f}%")

            if demo:
                st.caption("⚠ DEMO MODE - this is a placeholder prediction, not a real trained result.")

    st.divider()
    st.subheader("KL Grading Scale")
    for i, (grade, name, desc) in KL_LABELS.items():
        st.write(f"**{grade} ({name})** — {desc}")


# ----------------------------------------------------------------
# ENTRY POINT
# ----------------------------------------------------------------
# - Running "python knee_oa_app.py --train"     -> trains the model
# - Running "python knee_oa_app.py path/to.jpg" -> predicts one image
# - Running "streamlit run knee_oa_app.py"      -> launches the website
# ----------------------------------------------------------------
if __name__ == "__main__":
    if "streamlit" in sys.modules or "streamlit.runtime" in sys.modules:
        # Launched via `streamlit run knee_oa_app.py`
        run_website()
    elif len(sys.argv) > 1 and sys.argv[1] == "--train":
        train_model()
    elif len(sys.argv) > 1:
        predict_from_path(sys.argv[1])
    else:
        print(__doc__)
