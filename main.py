import cv2
import mediapipe as mp
import tkinter as tk
from tkinter import ttk
from utils.filters import apply_blur, apply_edges, apply_brightness
from utils.colors_mods import change_hue, adjust_saturation
import os

# MediaPipe Hands Configuration
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Haarcascade Path
HAAR_PATH = "utils/haarcascade_frontalface_default.xml"

if not os.path.exists(HAAR_PATH):
    raise FileNotFoundError(f"El archivo {HAAR_PATH} no existe. Asegúrate de que la ruta es correcta.")
else:
    print(f"Archivo Haarcascade disponible en {HAAR_PATH}")

# Filters and Color Bounds
FILTERS = ["Original", "Blur", "Median Blur", "Canny Edges", "Laplacian", "Brighten", "Hue", "Saturation"]
COLOR_BOUNDS = {
    "Rojo": [(0, 120, 70), (10, 255, 255)],
    "Verde": [(36, 100, 100), (86, 255, 255)],
    "Azul": [(94, 80, 2), (126, 255, 255)],
}

# Global Variables
current_filter = 0
hue_value = 0
saturation_scale = 1.0
brightness_scale = 1.0
paused = False
last_gesture = None

# Utility Function: Apply Custom Filters
def apply_custom_filters(frame, filter_type):
    if filter_type == "Median Blur":
        return cv2.medianBlur(frame, 5)
    elif filter_type == "Canny Edges":
        return cv2.Canny(frame, 100, 200)
    elif filter_type == "Laplacian":
        return cv2.convertScaleAbs(cv2.Laplacian(frame, cv2.CV_64F))
    return frame

# Gesture Detection Functions
def detect_gesture(hand_landmarks):
    thumb_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP]
    index_tip = hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP]
    thumb_index_dist = abs(thumb_tip.x - index_tip.x)

    if thumb_index_dist < 0.03:
        return "Pulgar Arriba"
    elif thumb_tip.y < index_tip.y:
        return "Cambiar Filtro"
    elif thumb_tip.y > index_tip.y:
        return "Pausar/Reanudar"
    return None

def detect_open_hand(hand_landmarks):
    fingers_extended = [
        hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_TIP].y <
        hand_landmarks.landmark[mp.solutions.hands.HandLandmark.THUMB_IP].y,
        hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_TIP].y <
        hand_landmarks.landmark[mp.solutions.hands.HandLandmark.INDEX_FINGER_DIP].y,
        hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_TIP].y <
        hand_landmarks.landmark[mp.solutions.hands.HandLandmark.MIDDLE_FINGER_DIP].y,
        hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_TIP].y <
        hand_landmarks.landmark[mp.solutions.hands.HandLandmark.RING_FINGER_DIP].y,
        hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_TIP].y <
        hand_landmarks.landmark[mp.solutions.hands.HandLandmark.PINKY_DIP].y,
    ]
    return all(fingers_extended)

# Video Processing with Gesture Control
def process_video_with_gestures():
    global current_filter, hue_value, saturation_scale, brightness_scale, paused, last_gesture

    cap = cv2.VideoCapture(1)  # Camera index (change to 1 if needed)
    if not cap.isOpened():
        print("Error al abrir la cámara")
        return

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el cuadro de video")
            break

        # Convert frame to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Detect hand landmarks
        result = hands.process(rgb_frame)
        gesture_text = "Sin Gestos Detectados"

        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

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

        # Apply the current filter if not paused
        if not paused:
            if FILTERS[current_filter] == "Blur":
                frame = apply_blur(frame)
            elif FILTERS[current_filter] == "Brighten":
                frame = apply_brightness(frame, alpha=brightness_scale)
            elif FILTERS[current_filter] == "Hue":
                frame = change_hue(frame, hue_value)
            elif FILTERS[current_filter] == "Saturation":
                frame = adjust_saturation(frame, saturation_scale)
            else:
                frame = apply_custom_filters(frame, FILTERS[current_filter])

        # Display filter and gesture
        cv2.putText(frame, f"Filtro Activo: {FILTERS[current_filter]}", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
        cv2.putText(frame, gesture_text, (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        # Show video
        cv2.imshow("Procesamiento en Tiempo Real con Gestos", frame)

        # Exit on 'q'
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

# GUI Functions
def update_filter(filter_index):
    global current_filter
    current_filter = filter_index

def update_hue(value):
    global hue_value
    hue_value = int(value)

def update_saturation(value):
    global saturation_scale
    saturation_scale = float(value)

def update_brightness(value):
    global brightness_scale
    brightness_scale = float(value)

def toggle_pause():
    global paused
    paused = not paused

# Create GUI
def create_gui():
    root = tk.Tk()
    root.title("Control de Filtros con Gestos - Procesamiento de Imágenes")
    root.geometry("400x500")

    # Dropdown to select filter
    ttk.Label(root, text="Seleccione un filtro:").pack(pady=10)
    filter_combo = ttk.Combobox(root, values=FILTERS, state="readonly")
    filter_combo.pack()
    filter_combo.bind("<<ComboboxSelected>>", lambda e: update_filter(filter_combo.current()))

    # Sliders for adjustments
    ttk.Label(root, text="Ajustar Tonalidad (Hue):").pack(pady=10)
    hue_slider = tk.Scale(root, from_=0, to=180, orient="horizontal", command=update_hue)
    hue_slider.pack()

    ttk.Label(root, text="Ajustar Saturación:").pack(pady=10)
    saturation_slider = tk.Scale(root, from_=0.5, to=2.0, resolution=0.1, orient="horizontal", command=update_saturation)
    saturation_slider.pack()

    ttk.Label(root, text="Ajustar Brillo:").pack(pady=10)
    brightness_slider = tk.Scale(root, from_=0.5, to=2.0, resolution=0.1, orient="horizontal", command=update_brightness)
    brightness_slider.pack()

    # Buttons
    ttk.Button(root, text="Iniciar Video con Gestos", command=process_video_with_gestures).pack(pady=10)
    ttk.Button(root, text="Pausar/Reanudar", command=toggle_pause).pack(pady=10)
    ttk.Button(root, text="Salir", command=root.quit).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
