import cv2
import numpy as np
import tensorflow as tf
from mediapipe.python.solutions.holistic import Holistic
from helpers import mediapipe_detection, draw_keypoints
from gtts import gTTS
import pygame
import io

MODEL_PATH = r'C:\Users\cseba\OneDrive\Escritorio\Proyecto_final\best_model.keras'

model = tf.keras.models.load_model(MODEL_PATH)

gestures = ['gracias','hola','ok','te amo']

CONFIDENCE_THRESHOLD = 0.8
# Tamaño de la cola para suavizar las predicciones
PREDICTION_QUEUE_SIZE = 5
GESTURE_DISPLAY_TIME = 30  # Número de frames para mantener el gesto en pantalla

def keypoints_to_image(keypoints, width=224, height=224):
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(0, len(keypoints), 3):
        x = int(keypoints[i] * width)
        y = int(keypoints[i+1] * height)
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    return image

def preprocess_results(results):
    keypoints = []
    if results.left_hand_landmarks:
        for landmark in results.left_hand_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])
    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])
    
    if len(keypoints) == 0:
        return None  
    
    keypoints = np.array(keypoints)
    image = keypoints_to_image(keypoints)
    
    # Dimensiones de imagenes
    image = cv2.resize(image, (224, 224))  
    image = np.expand_dims(image, axis=0)  # Añadiendo una dimensión de batch
    return image

def smooth_predictions(predictions, queue_size):
    if len(predictions) < queue_size:
        return predictions[-1]  # Retorna la ultima predicciun si hay pocas en la cola
    return max(set(predictions), key=predictions.count)  # Predicciun mus comun en la cola

def detectar_gestos_en_tiempo_real():
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)
        
        # Aumentar el tamaño de la ventana
        video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)  # Ancho de la ventana
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)  # Altura de la ventana
        
        last_gesture = None
        prediction_queue = []
        gesture_counter = 0
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Voltear la imagen para la vista espejo para mejorar la experiencia de usuario
            results = mediapipe_detection(frame, holistic_model)

            # Filtrar puntos clave solo de las manos
            image = preprocess_results(results)
            
            if image is None:
                gesture = 'No se detectaron manos'
            else:
                prediction = model.predict(image)
                gesture_index = np.argmax(prediction, axis=1)[0]
                confidence = np.max(prediction, axis=1)[0]

                # Umbral de confianza para la predicción
                if confidence < CONFIDENCE_THRESHOLD:
                    gesture = 'Gesto no identificado'
                else:
                    gesture = gestures[gesture_index]
                    prediction_queue.append(gesture)
                    reproducir_palabra(gesture)

                    # Suavizar la predicción final(esto ayuda a la prediccion del gesto)
                    if len(prediction_queue) > PREDICTION_QUEUE_SIZE:
                        prediction_queue.pop(0)
                    
                    smoothed_gesture = smooth_predictions(prediction_queue, PREDICTION_QUEUE_SIZE)

                    if smoothed_gesture != last_gesture:
                        gesture_counter = GESTURE_DISPLAY_TIME
                        last_gesture = smoothed_gesture
                    else:
                        gesture_counter -= 1

                    # Mantener el gesto en pantalla si el contador no ha expirado
                    if gesture_counter > 0:
                        gesture = last_gesture
                    else:
                        gesture = 'No se detectaron manos'
            
            cv2.putText(frame, f'Gesto: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            draw_keypoints(frame, results)
            cv2.imshow('Handspeak', frame)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

#Utilizando pygame para reproducir la palabra o letra 
def reproducir_palabra(texto):
    tts = gTTS(text=texto, lang='es')
    
    mp3_fp = io.BytesIO()
    tts.write_to_fp(mp3_fp)
    mp3_fp.seek(0)
    
    pygame.mixer.init()
    pygame.mixer.music.load(mp3_fp, 'mp3')
    pygame.mixer.music.play()
    
    while pygame.mixer.music.get_busy():
        continue 

if __name__ == "__main__":
    detectar_gestos_en_tiempo_real()
