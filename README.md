# ✏️ MNIST Digit Recogniser

> *Draw a digit. Watch a neural network think.*

[![GitHub](https://img.shields.io/badge/GitHub-mnist--digit--recogniser-black?logo=github)](https://github.com/ruthsepulveda/mnist-digit-recogniser)

---

## Description

Interactive application that uses a Convolutional Neural Network (CNN)
trained on the MNIST dataset to recognise handwritten digits in real time.
The user draws a digit on a canvas with the mouse and the model predicts
instantly, showing the confidence score, top 3 predictions, and the
activation maps of each convolutional layer.

---

## Demo

![Demo](assets/demo.gif)

---

## App modes

### Normal mode
Draw any digit from 0 to 9 and the model predicts in real time with:
- Predicted digit with confidence score
- Top 3 predictions with probability breakdown
- CNN activation map visualisation showing what each layer detects

### Challenge mode
The app shows a target digit. Draw it and the model verifies whether
it recognised it correctly. A streak counter tracks your best run.

---

## Model

| Feature | Detail |
|---|---|
| Architecture | CNN (Convolutional Neural Network) |
| Framework | Keras + TensorFlow |
| Dataset | MNIST (60,000 training / 10,000 test images) |
| Test accuracy | 99.21% |
| Test loss | 0.0244 |
| Trainable parameters | 225,034 |
| Random seed | 42 |

### Architecture

    Input (28x28x1)
        ↓
    Conv2D(32, 3x3, ReLU)
        ↓
    MaxPooling2D(2x2)
        ↓
    Conv2D(64, 3x3, ReLU)
        ↓
    MaxPooling2D(2x2)
        ↓
    Flatten → 1,600 values
        ↓
    Dense(128, ReLU)
        ↓
    Dropout(0.5)
        ↓
    Dense(10, Softmax)

---

## Preprocessing controls

The sidebar exposes 4 controls that affect how the drawing is
processed before reaching the model:

| Control | Type | Effect |
|---|---|---|
| Stroke width | Slider | Changes the brush thickness on the canvas |
| Binarization threshold | Slider | Determines which pixels activate as white |
| Gaussian blur | Checkbox | Smooths the stroke to reduce noise |
| Invert colours | Toggle | Switches between black/white background |

---

## Activation maps

After each prediction the app visualises what the two convolutional
layers detected in the drawing:

- **Layer 1:** edges and lines
- **Layer 2:** complex patterns and digit parts
- **Input (28x28):** the actual image the model received

This makes the model's decision process interpretable and visible
to any user, not just data scientists.

---

## Run locally with Docker

    cd app/
    docker build -t mnist-app .
    docker run -p 7860:7860 -d mnist-app

Open http://localhost:7860 in your browser.

---

## Run locally with Poetry

    cd app/
    poetry install
    poetry run streamlit run main.py

---

## Project structure

    mnist-digit-recogniser/
    ├── parte0/
    │   ├── train.py
    │   ├── pyproject.toml
    │   ├── poetry.lock
    │   └── README.md
    ├── app/
    │   ├── Dockerfile
    │   ├── main.py
    │   ├── model.keras
    │   ├── pyproject.toml
    │   ├── poetry.lock
    │   └── README.md
    ├── assets/
    │   └── demo.gif
    └── README.md

---

## Tech stack

- Python 3.12
- Keras + TensorFlow
- Streamlit
- streamlit-drawable-canvas
- OpenCV
- Pillow
- Poetry
- Docker

---

## About

Project developed as part of the Applied Machine Learning Diploma
(UC Chile), Plataformas para ML course.

The activation map visualisation goes beyond the course requirements,
showing how each convolutional layer processes the input and where
the model focuses its attention when making a prediction.

---

*Dataset: MNIST, available directly in Keras. No external download required.*

