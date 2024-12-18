# Proyecto Clasificador de Imágenes (Gatos vs Perros)

Este proyecto es un clasificador de imágenes que distingue entre gatos y perros utilizando una red neuronal convolucional (CNN) implementada con TensorFlow y Keras.

## Estructura del Proyecto

- `src/train_model.py`: Script principal para entrenar y evaluar el modelo.
- `data/`: Directorio que contiene los datos de entrenamiento, validación y prueba.
    - `train/`: Imágenes de entrenamiento.
    - `validation/`: Imágenes de validación.
    - `test/`: Imágenes de prueba.
- `models/`: Directorio donde se guardará el modelo entrenado.
- `results/`: Directorio donde se guardarán los resultados, como las gráficas de entrenamiento.

## Requisitos

- Python 3.x
- TensorFlow
- Matplotlib

Puedes instalar las dependencias necesarias utilizando `pip` 

# Cómo ejecutar el script con el entorno virtual

## Requisitos previos

- Python 3.x
- `virtualenv` instalado

## Pasos para ejecutar el script

1. Clona el repositorio:
    ```sh
    git clone {repositorio_url}
    ```

2. Navega al directorio del proyecto:
    ```sh
    cd {nombre_del_proyecto}
    ```

3. Crea un entorno virtual:
    ```sh
    python3 -m venv venv
    ```

4. Activa el entorno virtual:
    ```sh
    source venv/bin/activate
    ```

5. Instala las dependencias:
    ```sh
    pip install -r requirements.txt
    ```

6. Ejecuta el script:
    ```sh
    python {nombre_del_script}.py
    ```

7. Para desactivar el entorno virtual:
    ```sh
    deactivate
    ```

## Reemplazar

- `{repositorio_url}`: URL del repositorio.
- `{nombre_del_proyecto}`: Nombre del directorio del proyecto.
- `{nombre_del_script}`: Nombre del archivo del script.