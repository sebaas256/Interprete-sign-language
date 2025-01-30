Proyecto de Reconocimiento de Gestos en Lenguaje de Señas-Handspeak

Este proyecto tiene como objetivo reconocer gestos de lenguaje de señas utilizando una camara en tiempo real, basado en la deteccion de puntos clave de las manos. Los gestos son procesados por modelos de aprendizaje profundo para identificar letras o palabras, y la salida puede ser reproducida como texto o en forma de audio.

Contenidos del Proyecto

    Modelos Entrenados: Los modelos utilizados en este proyecto son redes neuronales entrenadas para reconocer distintos gestos. Cada modelo se entrena con imagenes capturadas manualmente y almacenadas en carpetas organizadas por gestos.

    Captura de Imagenes (capture_samples.py): Script para capturar imagenes de gestos en tiempo real utilizando MediaPipe. Las imágenes se almacenan en directorios especificos para cada gesto.

    Entrenamiento de los modelos (modelo_aprendizaje.py): Este script esta programado para poder crear nuestra red neuronal y poderla entrenar con cualquier gesto que quieras, ten encuenta que para el entrenamiento nosotros utilizamos un modelo preentrenado (MobileNetV2) que funciona para el reconocimiento de imagenes, este puedes cambiarlo por cualquiera que quieras.

    Reentrenamiento de modelos (reentrenamiento.py): Este script funciona para poder reentrenar redes neuronales con imagenes nuevas ya sea de cualquier mismo gesto o algun nuevo gesto que quieras agregar al igual que en el entrenamiento utilizamos MobileNetV2.

    Deteccion en Tiempo Real (main.py): Este archivo es el encargado de procesar el video de la camara, reconocer los gestos y mostrar la salida en tiempo real. Utiliza modelos preentrenados para realizar la prediccion de gestos, convertirlos en texto y voz.

    Ayudas Visuales (helpers.py): Proporciona funciones para visualizar los puntos clave de las manos, dibujar barras de confianza, y mostrar texto en la pantalla durante la deteccion de gestos.


Requisitos

Para ejecutar el proyecto, necesitaras instalar las siguientes dependencias:

    tensorflow
    mediapipe
    opencv-python
    pygame
    gtts

Puedes instalar las dependencias con el siguiente comando:

    ""pip install tensorflow mediapipe opencv-python pygame gtts""


Personalizacion

    Modificacion de Gestos: Puedes modificar los gestos que el sistema reconoce ajustando las listas de gestos en los scripts y capturando nuevas imagenes.
    Modelos: El proyecto soporta la carga de multiples modelos para mejorar la precision en la deteccion de gestos.
