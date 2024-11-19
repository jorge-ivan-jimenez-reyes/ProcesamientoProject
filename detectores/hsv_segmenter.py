import cv2

def segment_by_color(frame, lower_color, upper_color):
    """
    Segmenta objetos de un color específico en un frame.
    """
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)  # Convertir a HSV
    mask = cv2.inRange(hsv_frame, lower_color, upper_color)  # Crear una máscara
    segmented_frame = cv2.bitwise_and(frame, frame, mask=mask)  # Aplicar máscara
    return segmented_frame
