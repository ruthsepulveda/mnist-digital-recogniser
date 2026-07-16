# App: MNIST Digit Recogniser

## Description
Streamlit application that uses a CNN trained on MNIST to recognise
handwritten digits. The user draws a digit on an interactive canvas
and the model predicts in real time.

## Requirements
- Docker Desktop installed and running

## Run with Docker

1. Build the image

    docker build -t mnist-app .

2. Run the container

    docker run -p 7860:7860 -d mnist-app

3. Open your browser at http://localhost:7860

## Run with Poetry (without Docker)

1. Install dependencies

    poetry install

2. Run the app

    poetry run streamlit run main.py

## Structure

    app/
    ├── Dockerfile
    ├── main.py
    ├── model.keras
    ├── pyproject.toml
    ├── poetry.lock
    └── README.md

## App features
- Interactive canvas to draw digits with the mouse
- Real-time prediction with confidence score
- Top 3 predictions with probability breakdown
- CNN activation map visualisation showing what each
  convolutional layer detects in the drawing
- Challenge mode: the app shows a target digit, the user
  draws it, and the model verifies it with streak tracking
- Preprocessing controls: stroke width, binarization threshold,
  gaussian blur, and colour inversion

