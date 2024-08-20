import cv2
import numpy as np
import tensorflow as tf
from mediapipe.python.solutions.holistic import Holistic
from helpers import mediapipe_detection, draw_keypoints

# Ruta al modelo entrenado
MODEL_PATH = r'C:\Users\lilia\OneDrive\Escritorio\modelo_lstm_lsp-main\modelo_lstm_lsp-main\modelo_gestos.h5'

# Carga el modelo entrenado
model = tf.keras.models.load_model(MODEL_PATH)

# Lista de nombres de gestos en el mismo orden en que se entrenó el modelo
gestures = ['ok', 'gracias', 'hola']

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
    return np.expand_dims(image, axis=0)  # Añade una dimensión de batch

def detectar_gestos_en_tiempo_real():
    """Captura video en tiempo real y detecta gestos usando el modelo de MediaPipe y TensorFlow."""
    with Holistic() as holistic_model:
        video = cv2.VideoCapture(0)
        
        while video.isOpened():
            ret, frame = video.read()
            if not ret:
                break
            
            results = mediapipe_detection(frame, holistic_model)
            
            # Preprocesar los resultados para el modelo
            image = preprocess_results(results)
            
            if image is None:
                gesture = 'No se detectaron manos'
            else:
                prediction = model.predict(image)
                gesture_index = np.argmax(prediction, axis=1)[0]
                confidence = np.max(prediction, axis=1)[0]

                # Mostrar el nombre del gesto y la confianza
                if confidence < 0.5:  # Ajusta el umbral según sea necesario
                    gesture = 'Gesto no identificado'
                else:
                    gesture = gestures[gesture_index]
            
            cv2.putText(frame, f'Gesto: {gesture}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
            draw_keypoints(frame, results)
            cv2.imshow('Detección de Gestos', frame)
            
            # Presiona 'q' para salir del bucle
            if cv2.waitKey(10) & 0xFF == ord('q'):
                break

        video.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    detectar_gestos_en_tiempo_real()
