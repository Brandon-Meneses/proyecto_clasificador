import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import os


# Configuración inicial
train_dir = "../data/train"
val_dir = "../data/validation"
test_dir = "../data/test"

img_size = (150, 150)
batch_size = 32

# Cargamos los datos desde el directorio, creando "tf.data.Dataset"
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    train_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='binary',
    shuffle=True
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    val_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='binary',
    shuffle=True
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    test_dir,
    image_size=img_size,
    batch_size=batch_size,
    label_mode='binary',
    shuffle=False
)

# Opcional: Aumentación de datos
data_augmentation = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.1),
    layers.RandomZoom(0.1),
])

# Normalización de píxeles
normalization_layer = layers.Rescaling(1./255)

# Creación del modelo (CNN simple)
model = keras.Sequential([
    # Aumentación (sólo en entrenamiento)
    data_augmentation,
    # Normalización
    normalization_layer,
    
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(150,150,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(128, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    # Aplanamos las características extraídas por las convoluciones
    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.5),  # Regularización para evitar sobreajuste
    layers.Dense(1, activation='sigmoid')  # Salida binaria (gato vs perro)
])

model.compile(optimizer='adam',
              loss='binary_crossentropy',
              metrics=['accuracy'])

model.summary()

# Crear carpetas en la raíz del proyecto
root_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
models_dir = os.path.join(root_dir, "models")
results_dir = os.path.join(root_dir, "results")

os.makedirs(models_dir, exist_ok=True)
os.makedirs(results_dir, exist_ok=True)

# Entrenamiento del modelo
epochs = 10
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=epochs
)

# Evaluación en el conjunto de prueba
loss, accuracy = model.evaluate(test_ds)
print("Exactitud en test:", accuracy)

# Guardar el modelo
model.save("models/modelo_gatos_perros.h5")
print("Modelo guardado en 'models/modelo_gatos_perros.h5'")

# Si quieres graficar la evolución del entrenamiento:
import matplotlib.pyplot as plt

acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss_vals = history.history['loss']
val_loss_vals = history.history['val_loss']

plt.figure(figsize=(8, 4))
plt.subplot(1, 2, 1)
plt.plot(acc, label='Entrenamiento')
plt.plot(val_acc, label='Validación')
plt.title('Exactitud')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(loss_vals, label='Entrenamiento')
plt.plot(val_loss_vals, label='Validación')
plt.title('Pérdida')
plt.legend()

os.makedirs("models", exist_ok=True)
os.makedirs("results", exist_ok=True)



# Guardar la gráfica
plt.savefig(os.path.join(results_dir, "training_plot.png"))
print(f"Gráfica guardada en '{os.path.join(results_dir, 'training_plot.png')}'")

plt.show()
