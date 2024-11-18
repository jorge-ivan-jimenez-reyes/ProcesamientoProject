import cv2
from detectores.hear_detectors import detect_objects
from detectores.hsv_segmenter import segment_by_color
from utils.filters import apply_blur, apply_edges, apply_brightness
from utils.colors_mods import change_hue, adjust_saturation

import os

# Ruta al archivo Haarcascade
HAAR_PATH = "utils/haarcascade_frontalface_default.xml"

# Verificar si el archivo Haarcascade existe
if not os.path.exists(HAAR_PATH):
    raise FileNotFoundError(f"El archivo {HAAR_PATH} no existe. Asegúrate de que la ruta es correcta.")
else:
    print(f"Archivo Haarcascade disponible en {HAAR_PATH}")

def main():
    # Captura de video desde la cámara
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error al abrir la cámara")
        return

    # Variables iniciales para filtros y configuración
    current_filter = "none"
    lower_color = (0, 120, 70)  # Límites de color en HSV (rojo como ejemplo)
    upper_color = (10, 255, 255)
    hue_value = 0
    scale_saturation = 1.0

    while True:
        ret, frame = cap.read()
        if not ret:
            print("Error al capturar el cuadro de video")
            break

        # Detección de objetos con Haarcascade
        try:
            frame = detect_objects(frame, HAAR_PATH)
        except Exception as e:
            print(f"Error en la detección de objetos: {e}")

        # Segmentación por color
        segmented_frame = segment_by_color(frame, lower_color, upper_color)

        # Aplicación de filtros según el filtro actual
        if current_filter == "blur":
            frame = apply_blur(segmented_frame)
        elif current_filter == "edges":
            frame = apply_edges(segmented_frame)
        elif current_filter == "brighten":
            frame = apply_brightness(segmented_frame)
        elif current_filter == "hue":
            frame = change_hue(segmented_frame, hue_value)
        elif current_filter == "saturation":
            frame = adjust_saturation(segmented_frame, scale_saturation)

        # Mostrar el resultado en tiempo real
        cv2.imshow("Procesamiento en Tiempo Real", frame)

        # Controles del teclado
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):  # Salir
            break
        elif key == ord('b'):  # Blur
            current_filter = "blur"
        elif key == ord('e'):  # Detección de bordes
            current_filter = "edges"
        elif key == ord('r'):  # Brillo
            current_filter = "brighten"
        elif key == ord('h'):  # Cambiar tonalidad (hue)
            current_filter = "hue"
            hue_value = (hue_value + 10) % 180
        elif key == ord('s'):  # Aumentar saturación
            current_filter = "saturation"
            scale_saturation += 0.1

    # Liberar recursos
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
