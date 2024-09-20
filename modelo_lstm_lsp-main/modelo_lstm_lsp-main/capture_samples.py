import os
import cv2
import numpy as np
from mediapipe.python.solutions.holistic import Holistic
from datetime import datetime

# Configuración de carpeta para guardar imágenes
GESTOS_DIR = 'gestos'
gestos = ['A']  # Lista de gestos que quieres capturar

# Crear carpetas para cada gesto
for gesto in gestos:
    path = os.path.join(GESTOS_DIR, gesto)
    os.makedirs(path, exist_ok=True)

def preprocesar_frame(frame, img_size=(224, 224)):
    imagen = cv2.resize(frame, img_size)
    return imagen

def mediapipe_detection(image, model):
    # Convertir la imagen de BGR a RGB para MediaPipe
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_rgb.flags.writeable = False  # Hacer la imagen no modificable para mejorar la eficiencia
    results = model.process(image_rgb)  # Realizar la detección
    image_rgb.flags.writeable = True   # Hacer la imagen modificable nuevamente
    return results

def there_hand(results):
    return results.left_hand_landmarks or results.right_hand_landmarks

def capturar_gestos():
    cap = cv2.VideoCapture(0)
    gesture_idx = 0
    capturando = False
    frames = []

    with Holistic() as holistic_model:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)  # Voltear horizontalmente para mejor experiencia de usuario
            image = frame.copy()
            results = mediapipe_detection(frame, holistic_model)
            
            # Mostrar instrucciones
            cv2.putText(image, f'Capturando gesto: {gestos[gesture_idx]}', (10, 50),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            if there_hand(results):
                if not capturando:
                    print("Manos detectadas, comenzando captura...")
                    capturando = True
                
                hoy = datetime.now().strftime('%Y%m%d_%H%M%S%f')
                img_path = os.path.join(GESTOS_DIR, gestos[gesture_idx], f'{hoy}.jpg')
                procesado = preprocesar_frame(frame)
                frames.append(procesado)
                cv2.imwrite(img_path, procesado)
                print(f'Frame capturado y guardado en {img_path}')
            else:
                if capturando:
                    print("Manos no detectadas, deteniendo captura...")
                    capturando = False
                    frames = []  # Reinicia los frames

            # Mostrar imagen procesada
            cv2.imshow('Captura de Gestos', image)
            
            key = cv2.waitKey(10) & 0xFF
            if key == ord('n'):
                gesture_idx += 1
                if gesture_idx >= len(gestos):
                    print('Captura completada.')
                    break
                print(f'Preparado para capturar el siguiente gesto: {gestos[gesture_idx]}')
            elif key == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    capturar_gestos()
