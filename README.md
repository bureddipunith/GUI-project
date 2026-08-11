# Knee Osteoarthritis Detection using Deep Learning

Predicts the KL (Kellgren-Lawrence) grade of knee osteoarthritis from an
X-ray image, using a CNN built with TensorFlow.

**Tech used:** TensorFlow, NumPy, Pandas, Streamlit

Everything (model, training, prediction, website) is in one file:
`knee_oa_app.py`

## KL Grading Scale

| Grade | Name | Description |
|-------|------|-------------|
| 0 | Normal | No signs of knee osteoarthritis |
| 1 | Doubtful | Very early signs; doubtful joint narrowing or small bone spurs |
| 2 | Mild | Definite bone spurs and slight joint space narrowing |
| 3 | Moderate | Clear joint space narrowing, multiple bone spurs, some bone changes |
| 4 | Severe | Large bone spurs, severe joint space narrowing, major bone damage |

## Setup

```bash
pip install -r requirements.txt
```

## Step 1: Get a dataset

Download a public knee X-ray dataset, e.g.:
- [Knee Osteoarthritis Dataset with Severity Grading (Kaggle)](https://www.kaggle.com/datasets/shashwatwork/knee-osteoarthritis-dataset-with-severity)

Arrange it like this:

```
dataset/
  images/
    img001.png
    img002.png
    ...
  labels.csv
```

`labels.csv` needs two columns:

```csv
filename,label
img001.png,0
img002.png,2
img003.png,4
```

(`label` = the KL grade, 0 to 4)

## Step 2: Train the model

```bash
python knee_oa_app.py --train
```

Trains the CNN and saves it to `model/kl_model.h5`.

## Step 3: Test a single image (optional)

```bash
python knee_oa_app.py path/to/xray.jpg
```

## Step 4: Run the website

```bash
streamlit run knee_oa_app.py
```

Opens a website — upload an X-ray, click **Predict**, see the KL grade.

## Note

Until you complete Steps 1–2, the website runs in **DEMO MODE**: it works
end-to-end, but the predicted grade is a random placeholder, clearly
labeled on the page. This lets you test and demo the app before the model
is trained.

## How the CNN works (for viva/interview explanation)

1. **3 Conv2D + MaxPooling blocks** — each detects increasingly complex
   patterns in the X-ray (edges → textures → joint/bone-spur shapes)
2. **Flatten** — turns 2D feature maps into a 1D vector
3. **Dense(128) + Dropout** — combines learned features, dropout reduces overfitting
4. **Dense(5, softmax)** — output layer, one probability per KL grade

Trained with the Adam optimizer and sparse categorical crossentropy loss
(labels are integers 0–4, not one-hot encoded).

## Author

Poornima Reddy — Final Year Project
