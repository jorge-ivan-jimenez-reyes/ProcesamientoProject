import cv2

def apply_blur(frame):
    return cv2.GaussianBlur(frame, (15, 15), 0)

def apply_edges(frame):
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    return cv2.Canny(gray_frame, 50, 150)

def apply_brightness(frame, alpha=1.2, beta=50):
    return cv2.convertScaleAbs(frame, alpha=alpha, beta=beta)
