import numpy as np
import keras
from keras import layers
import shutil
import os

# ── Semilla aleatoria para reproducibilidad ───────────────────────────────────
keras.utils.set_random_seed(42)

# ── 1. Cargar el dataset ──────────────────────────────────────────────────────
# MNIST viene incorporado en Keras, no requiere descarga externa
(x_train, y_train), (x_test, y_test) = keras.datasets.mnist.load_data()

print(f"Imágenes de entrenamiento: {x_train.shape}")
print(f"Imágenes de test: {x_test.shape}")

# ── 2. Preprocesamiento ───────────────────────────────────────────────────────
# Agregar dimensión de canal (requerido por Conv2D) y normalizar a rango 0-1
x_train = x_train.reshape(-1, 28, 28, 1).astype("float32") / 255.0
x_test = x_test.reshape(-1, 28, 28, 1).astype("float32") / 255.0

# ── 3. Arquitectura CNN ───────────────────────────────────────────────────────
# Red neuronal convolucional con dos bloques conv+pooling y un clasificador MLP
model = keras.Sequential([
    # Bloque convolucional 1: detecta patrones simples (bordes, líneas)
    layers.Conv2D(32, (3, 3), activation="relu", input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),

    # Bloque convolucional 2: detecta patrones complejos (curvas, partes de dígitos)
    layers.Conv2D(64, (3, 3), activation="relu"),
    layers.MaxPooling2D((2, 2)),

    # Clasificador
    layers.Flatten(),
    layers.Dense(128, activation="relu"),
    layers.Dropout(0.5),  # Regularización: evita overfitting
    layers.Dense(10, activation="softmax")  # 10 clases, una por dígito
])

model.summary()

# ── 4. Compilar ───────────────────────────────────────────────────────────────
# Adam: optimizador adaptativo, buena opción general
# sparse_categorical_crossentropy: para etiquetas como enteros (0-9)
model.compile(
    optimizer="adam",
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

# ── 5. Entrenar ───────────────────────────────────────────────────────────────
# 10 epochs, batches de 128, 10% de los datos reservados para validación
history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=128,
    validation_split=0.1,
    verbose=1
)

# ── 6. Evaluar ────────────────────────────────────────────────────────────────
test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)
print(f"Test accuracy: {test_accuracy:.4f}")
print(f"Test loss: {test_loss:.4f}")

# ── 7. Guardar el modelo ──────────────────────────────────────────────────────
# Formato .keras: recomendado desde Keras 3, incluye arquitectura y pesos
model.save("model.keras")
print("Modelo guardado como model.keras")

# ── 8. Copiar a la carpeta app/ ───────────────────────────────────────────────
# El enunciado requiere que el modelo esté en app/ para ser empaquetado con Docker
os.makedirs("../app", exist_ok=True)
shutil.copy("model.keras", "../app/model.keras")
print("Modelo copiado a ../app/model.keras")
