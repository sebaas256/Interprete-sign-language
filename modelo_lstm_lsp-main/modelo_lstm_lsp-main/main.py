import cv2
import numpy as np
import tensorflow as tf
from mediapipe.python.solutions.holistic import Holistic
from helpers import mediapipe_detection, draw_keypoints, draw_progress_bar, display_text
from gtts import gTTS
import pygame
import io

# Ruta del modelo
MODEL_PATH = r'C:\Users\lilia\OneDrive\Escritorio\Proyecto_final\try_abecedario.keras'
model = tf.keras.models.load_model(MODEL_PATH)

# Gestos reconocidos por el modelo
#gestures = ['gracias', 'hola', 'ok', 'te amo']
gestures = ['A', 'B', 'C']


# Umbral de confianza
CONFIDENCE_THRESHOLD = 0.8
PREDICTION_QUEUE_SIZE = 5
GESTURE_DISPLAY_TIME = 30

def keypoints_to_image(keypoints, width=224, height=224):
    """Convierte los puntos clave en una imagen para alimentar al modelo."""
    image = np.zeros((height, width, 3), dtype=np.uint8)
    for i in range(0, len(keypoints), 3):
        x = int(keypoints[i] * width)
        y = int(keypoints[i+1] * height)
        cv2.circle(image, (x, y), 5, (0, 255, 0), -1)
    return image

def preprocess_results(results):
    """Preprocesa los resultados de MediaPipe para pasarlos al modelo."""
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
    
    image = cv2.resize(image, (224, 224))  
    image = np.expand_dims(image, axis=0)
    return image

def smooth_predictions(predictions, queue_size):
    """Suaviza las predicciones para reducir fluctuaciones."""
    if len(predictions) < queue_size:
        return predictions[-1]
    return max(set(predictions), key=predictions.count)

def detectar_gestos_en_tiempo_real():
    """Función principal para detectar gestos en tiempo real."""
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)

        # Configurar el tamaño de la ventana
        video.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        video.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        
        last_gesture = None
        prediction_queue = []
        gesture_counter = 0
        frame_counter = 0  # Contador de frames para el efecto de pulso
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)
            results = mediapipe_detection(frame, holistic_model)
            image = preprocess_results(results)
            
            if image is None:
                gesture = 'No se detectaron manos'
                confidence = 0.0
            else:
                prediction = model.predict(image)
                gesture_index = np.argmax(prediction, axis=1)[0]
                confidence = np.max(prediction, axis=1)[0]

                if confidence < CONFIDENCE_THRESHOLD:
                    gesture = 'Gesto no identificado'
                else:
                    gesture = gestures[gesture_index]
                    prediction_queue.append(gesture)

                    if len(prediction_queue) > PREDICTION_QUEUE_SIZE:
                        prediction_queue.pop(0)
                    
                    smoothed_gesture = smooth_predictions(prediction_queue, PREDICTION_QUEUE_SIZE)

                    if smoothed_gesture != last_gesture:
                        gesture_counter = GESTURE_DISPLAY_TIME
                        last_gesture = smoothed_gesture
                        reproducir_palabra(gesture)
                    else:
                        gesture_counter -= 1

                    if gesture_counter > 0:
                        gesture = last_gesture
                    else:
                        gesture = 'No se detectaron manos'

            # Mostrar texto del gesto y confianza en estilo futurista
            display_text(frame, gesture, confidence)
            
            # Dibujar barra de progreso según el nivel de confianza
            draw_progress_bar(frame, confidence)
            
            # Dibujar puntos clave con diseño futurista
            draw_keypoints(frame, results, frame_counter)
            frame_counter += 1  # Actualizar el contador para el efecto de pulso

            cv2.imshow('Handspeak', frame)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

# Utilizando pygame para reproducir la palabra o letra 
def reproducir_palabra(texto):
    """Reproduce la palabra o letra detectada usando gTTS y pygame."""
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
