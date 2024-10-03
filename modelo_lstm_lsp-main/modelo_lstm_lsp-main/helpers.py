import cv2
import mediapipe as mp
import math

# Inicializa los módulos de MediaPipe
mp_holistic = mp.solutions.holistic
mp_drawing = mp.solutions.drawing_utils

# Colores futuristas
HAND_COLOR = (0, 255, 0)  # Verde neón
POSE_COLOR = (192, 192, 192)  # Gris suave
TEXT_COLOR = (0, 255, 255)  # Amarillo brillante

# Fuente personalizada
FONT = cv2.FONT_HERSHEY_DUPLEX

def mediapipe_detection(image, model):
    """Procesa la imagen para obtener los resultados de MediaPipe."""
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = model.process(image_rgb)
    return results

def draw_keypoints(image, results, frame_counter):
    """Dibuja los puntos clave y conexiones en la imagen con un efecto de pulso futurista."""
    pulse_radius = int(5 + 2 * math.sin(frame_counter * 0.1))  # Oscilación del tamaño del círculo
    
    # Dibuja la pose (en color gris suave para no distraer demasiado)
    if results.pose_landmarks:
        mp_drawing.draw_landmarks(
            image, results.pose_landmarks, mp_holistic.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=POSE_COLOR, thickness=1, circle_radius=1),
            mp_drawing.DrawingSpec(color=POSE_COLOR, thickness=1, circle_radius=1)
        )
    
    # Dibuja las manos en verde neón con animación de pulso
    if results.left_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.left_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=HAND_COLOR, thickness=2, circle_radius=pulse_radius),
            mp_drawing.DrawingSpec(color=HAND_COLOR, thickness=2, circle_radius=pulse_radius)
        )
    
    if results.right_hand_landmarks:
        mp_drawing.draw_landmarks(
            image, results.right_hand_landmarks, mp_holistic.HAND_CONNECTIONS,
            mp_drawing.DrawingSpec(color=HAND_COLOR, thickness=2, circle_radius=pulse_radius),
            mp_drawing.DrawingSpec(color=HAND_COLOR, thickness=2, circle_radius=pulse_radius)
        )

def there_hand(results):
    """Verifica si hay al menos una mano detectada en los resultados."""
    return results.left_hand_landmarks is not None or results.right_hand_landmarks is not None

def extract_hand_keypoints(results):
    """Extrae solo los puntos clave de las manos para el modelo de gestos."""
    keypoints = []

    # Verifica si hay puntos clave de la mano izquierda
    if results.left_hand_landmarks:
        for landmark in results.left_hand_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])
    
    # Verifica si hay puntos clave de la mano derecha
    if results.right_hand_landmarks:
        for landmark in results.right_hand_landmarks.landmark:
            keypoints.extend([landmark.x, landmark.y, landmark.z])

    # Devuelve los puntos clave de las manos
    if keypoints:
        return keypoints
    else:
        return None  # Si no se detectan manos, retorna None

def display_text(frame, text, confidence):
    """Muestra el texto en la pantalla con estilo futurista."""
    text_display = f'{text} ({confidence * 100:.2f}%)'
    cv2.putText(frame, text_display, (10, 50), FONT, 1, TEXT_COLOR, 2, cv2.LINE_AA)

def draw_progress_bar(frame, confidence):
    """Dibuja una barra de progreso en la pantalla para mostrar el nivel de confianza."""
    bar_width = int(confidence * 400)  # Ajusta el tamaño según la confianza
    cv2.rectangle(frame, (10, 80), (10 + bar_width, 110), TEXT_COLOR, -1)
    cv2.rectangle(frame, (10, 80), (410, 110), TEXT_COLOR, 2)