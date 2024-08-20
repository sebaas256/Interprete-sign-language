import cv2
import mediapipe as mp

# Inicializa los módulos de MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

def mediapipe_detection(image, model):
    """Procesa la imagen para obtener los resultados de MediaPipe."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image_rgb)
    return results

def draw_keypoints(image, results):
    """Dibuja los puntos clave y conexiones en la imagen."""
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS)
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS)
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS)

def there_hand(results):
    """Verifica si hay al menos una mano detectada en los resultados."""
    return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None
