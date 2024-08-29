import cv2
import numpy as np
import tensorflow as tf
from mediapipe.python.solutions.holistic import Holistic
from helpers import mediapipe_detection, draw_keypoints

# Ruta al modelo entrenado
MODEL_PATH = r'C:\Users\cseba\OneDrive\Escritorio\Proyecto_final\best_model.keras'

# Carga el modelo entrenado
model = tf.keras.models.load_model(MODEL_PATH)

# Lista de nombres de gestos en el mismo orden en que se entrenó el modelo
gestures = ['gracias','hola','ok']

# Umbral de confianza
CONFIDENCE_THRESHOLD = 0.8
# Tamaño de la cola para suavizar las predicciones
PREDICTION_QUEUE_SIZE = 5

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
        return None  # No se detectaron puntos clave
    
    keypoints = np.array(keypoints)
    image = keypoints_to_image(keypoints)
    
    # Asegurarse de que la imagen tiene las dimensiones correctas
    image = cv2.resize(image, (224, 224))  # Asegurarse del tamaño correcto
    image = np.expand_dims(image, axis=0)  # Añade una dimensión de batch
    return image

def smooth_predictions(predictions, queue_size):
    if len(predictions) < queue_size:
        return predictions[-1]  # Retorna la última predicción si hay pocas en la cola
    return max(set(predictions), key=predictions.count)  # Predicción más común en la cola

def detectar_gestos_en_tiempo_real():
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)
        
        last_gesture = None
        prediction_queue = []
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            frame = cv2.flip(frame, 1)  # Voltear la imagen para la vista espejo
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

                    # Suavizar la predicción final
                    if len(prediction_queue) > PREDICTION_QUEUE_SIZE:
                        prediction_queue.pop(0)
                    
                    gesture = smooth_predictions(prediction_queue, PREDICTION_QUEUE_SIZE)
            
            # Evitar predicciones repetitivas
            if gesture != last_gesture:
                last_gesture = gesture
            
            cv2.putText(frame, f'Gesto: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            draw_keypoints(frame, results)
            cv2.imshow('Detección de Gestos', frame)
            
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detectar_gestos_en_tiempo_real()
