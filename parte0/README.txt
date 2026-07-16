# Part 0: Model Training

## Description
Trains a Convolutional Neural Network (CNN) on the Keras MNIST dataset
to classify handwritten digits (0-9). The trained model is saved as
model.keras and copied automatically to ../app/.

## Requirements
- Python 3.12
- Poetry

## Instructions

### 1. Install dependencies

    poetry install

### 2. Train the model

    poetry run python train.py

The script trains the model, displays the final accuracy and copies
model.keras to the ../app/ folder automatically.

## Model details

- Architecture: CNN with 2 convolutional blocks + Dense + Dropout
- Dataset: MNIST (60,000 training images / 10,000 test images)
- Test accuracy: 99.21%
- Test loss: 0.0244
- Trainable parameters: 225,034
- Random seed: 42 (for reproducibility)

