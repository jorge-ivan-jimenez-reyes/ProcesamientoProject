import cv2
import numpy as np

def segment_by_color(frame, lower_color, upper_color):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv_frame, lower_color, upper_color)
    segmented_frame = cv2.bitwise_and(frame, frame, mask=mask)
    return segmented_frame
