import cv2
import numpy as np

def change_hue(frame, hue_value):
    """
    Cambia la tonalidad (Hue) de un objeto segmentado.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[..., 0] = hue_value
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)

def adjust_saturation(frame, scale):
    """
    Ajusta la saturación de un objeto segmentado.
    """
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    hsv[..., 1] = np.clip(hsv[..., 1] * scale, 0, 255)
    return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
