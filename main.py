import cv2
import tkinter as tk
from tkinter import ttk
from detectores.hear_detectors import detect_objects
from detectores.hsv_segmenter import segment_by_color
from utils.filters import apply_blur, apply_edges, apply_brightness
from utils.colors_mods import change_hue, adjust_saturation
import os

# Ruta al archivo Haarcascade
HAAR_PATH = "utils/haarcascade_frontalface_default.xml"

if not os.path.exists(HAAR_PATH):
    raise FileNotFoundError(f"El archivo {HAAR_PATH} no existe. Asegúrate de que la ruta es correcta.")
else:
    print(f"Archivo Haarcascade disponible en {HAAR_PATH}")

# Configuración inicial
FILTERS = ["original", "blur", "edges", "brighten", "hue", "saturation"]
COLOR_BOUNDS = {
    "rojo": [(0, 120, 70), (10, 255, 255)],
    "verde": [(36, 100, 100), (86, 255, 255)],
    "azul": [(94, 80, 2), (126, 255, 255)],
}

# Variables globales
current_filter = 0
hue_value = 0
saturation_scale = 1.0
brightness_scale = 1.0
paused = False

def process_video_main_thread():
    global current_filter, hue_value, saturation_scale, brightness_scale, paused

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error al abrir la cámara")
        return

    while True:
        if paused:
            cv2.waitKey(1)
            continue

        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el cuadro de video")
            break

        try:
            # Detección de objetos
            frame = detect_objects(frame, HAAR_PATH)

            # Procesar para cada color definido
            for color_name, (lower, upper) in COLOR_BOUNDS.items():
                segmented_frame = segment_by_color(frame, lower, upper)

                if segmented_frame is None or segmented_frame.size == 0:
                    print(f"Segmentación fallida para el color {color_name}")
                    continue

                # Aplicar filtro actual
                if FILTERS[current_filter] == "blur":
                    segmented_frame = apply_blur(segmented_frame)
                elif FILTERS[current_filter] == "edges":
                    segmented_frame = apply_edges(segmented_frame)
                elif FILTERS[current_filter] == "brighten":
                    segmented_frame = apply_brightness(segmented_frame, alpha=brightness_scale)
                elif FILTERS[current_filter] == "hue":
                    segmented_frame = change_hue(segmented_frame, hue_value)
                elif FILTERS[current_filter] == "saturation":
                    segmented_frame = adjust_saturation(segmented_frame, saturation_scale)

                # Mostrar resultados segmentados
                try:
                    cv2.imshow(f"Segmentación {color_name}", segmented_frame)
                except cv2.error as e:
                    print(f"Error al mostrar la ventana de segmentación {color_name}: {e}")

            # Mostrar resultado combinado
            cv2.imshow("Procesamiento en Tiempo Real", frame)

        except Exception as e:
            print(f"Error en el procesamiento del video: {e}")

        # Salir con 'q'
        try:
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        except cv2.error as e:
            print(f"Error en cv2.waitKey: {e}")

    cap.release()
    cv2.destroyAllWindows()

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

def create_gui():
    root = tk.Tk()
    root.title("Control de Filtros - Procesamiento de Imágenes")
    root.geometry("400x500")

    # Etiqueta principal
    ttk.Label(root, text="Seleccione un filtro:").pack(pady=10)

    # Dropdown para seleccionar filtro
    filter_combo = ttk.Combobox(root, values=FILTERS, state="readonly")
    filter_combo.pack()
    filter_combo.bind("<<ComboboxSelected>>", lambda e: update_filter(filter_combo.current()))

    # Sliders para ajuste de hue, saturación y brillo
    ttk.Label(root, text="Ajustar Tonalidad (Hue):").pack(pady=10)
    hue_slider = tk.Scale(root, from_=0, to=180, orient="horizontal", command=update_hue)
    hue_slider.pack()

    ttk.Label(root, text="Ajustar Saturación:").pack(pady=10)
    saturation_slider = tk.Scale(root, from_=0.5, to=2.0, resolution=0.1, orient="horizontal", command=update_saturation)
    saturation_slider.pack()

    ttk.Label(root, text="Ajustar Brillo:").pack(pady=10)
    brightness_slider = tk.Scale(root, from_=0.5, to=2.0, resolution=0.1, orient="horizontal", command=update_brightness)
    brightness_slider.pack()

    # Botones de control
    ttk.Button(root, text="Iniciar Video", command=process_video_main_thread).pack(pady=10)
    ttk.Button(root, text="Pausar/Reanudar", command=toggle_pause).pack(pady=10)

    # Botón para salir
    ttk.Button(root, text="Salir", command=root.quit).pack(pady=10)

    root.mainloop()

if __name__ == "__main__":
    create_gui()
