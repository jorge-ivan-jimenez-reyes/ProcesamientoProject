import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from detectores.hear_detectors import detect_objects
from detectores.hsv_segmenter import segment_by_color
from utils.filters import apply_blur, apply_edges, apply_brightness
from utils.colors_mods import change_hue, adjust_saturation
import os

# MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Ruta al archivo Haarcascade
HAAR_PATH = "utils/haarcascade_frontalface_default.xml"

if not os.path.exists(HAAR_PATH):
    raise FileNotFoundError(f"El archivo {HAAR_PATH} no existe. Asegúrate de que la ruta es correcta.")
else:
    print(f"Archivo Haarcascade disponible en {HAAR_PATH}")

# Configuración inicial
FILTERS = ["Original", "Blur", "Edges", "Brighten", "Hue", "Saturation"]
COLOR_BOUNDS = {
    "Rojo": [(0, 120, 70), (10, 255, 255)],
    "Verde": [(36, 100, 100), (86, 255, 255)],
    "Azul": [(94, 80, 2), (126, 255, 255)],
}

# Variables globales
current_filter = 0
hue_value = 0
saturation_scale = 1.0
brightness_scale = 1.0
paused = False
last_gesture = None

def detect_gesture(hand_landmarks):
    """
    Detecta gestos simples de la mano basados en la posición de los puntos clave.
    """
    thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]

    # Distancia entre pulgar e índice (gesto de "pulgar arriba")
    thumb_index_dist = abs(thumb_tip.x - index_tip.x)

    if thumb_index_dist < 0.03:  # Pulgar e índice juntos
        return "Pulgar Arriba"
    elif thumb_tip.y < index_tip.y:  # Pulgar arriba
        return "Cambiar Filtro"
    elif thumb_tip.y > index_tip.y:  # Pulgar abajo
        return "Pausar/Reanudar"

    return None

def process_video_with_gestures():
    global current_filter, hue_value, saturation_scale, brightness_scale, paused, last_gesture

    cap = cv2.VideoCapture(1)
    if not cap.isOpened():
        print("Error al abrir la cámara")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el cuadro de video")
            break

        # Convertir la imagen a RGB para MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Procesar la detección de manos
        result = hands.process(rgb_frame)

        gesture_text = "Sin Gestos Detectados"

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )

                # Detectar gesto
                gesture = detect_gesture(hand_landmarks)
                if gesture:
                    gesture_text = f"Gesto: {gesture}"
                    last_gesture = gesture
                    if gesture == "Cambiar Filtro" and not paused:
                        current_filter = (current_filter + 1) % len(FILTERS)
                        print(f"Filtro actual: {FILTERS[current_filter]}")
                    elif gesture == "Pausar/Reanudar":
                        paused = not paused
                        print("Video pausado" if paused else "Video reanudado")

        # Aplicar filtro actual
        if not paused:
            if FILTERS[current_filter] == "Blur":
                frame = apply_blur(frame)
            elif FILTERS[current_filter] == "Edges":
                frame = apply_edges(frame)
            elif FILTERS[current_filter] == "Brighten":
                frame = apply_brightness(frame)
            elif FILTERS[current_filter] == "Hue":
                frame = change_hue(frame, hue_value)
            elif FILTERS[current_filter] == "Saturation":
                frame = adjust_saturation(frame, saturation_scale)

        # Mostrar el filtro activo y gesto detectado
        cv2.putText(frame, f"Filtro Activo: {FILTERS[current_filter]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, gesture_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # Mostrar resultado
        cv2.imshow("Procesamiento en Tiempo Real con Gestos", frame)

        # Salir con 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def create_gui():
    root = tk.Tk()
    root.title("Control de Filtros con Gestos - Procesamiento de Imágenes")
    root.geometry("400x500")

    # Botones de control
    ttk.Button(root, text="Iniciar Video con Gestos", command=process_video_with_gestures).pack(pady=10)

    # Botón para salir
    ttk.Button(root, text="Salir", command=root.quit).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
