import cv2

def detect_objects(frame, classifier_path):
    """
    Detecta objetos (por ejemplo, rostros) en un frame usando clasificadores Haar.
    """
    gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    classifier = cv2.CascadeClassifier(classifier_path)

    if classifier.empty():
        raise FileNotFoundError(f"No se pudo cargar el clasificador desde {classifier_path}")

    objects = classifier.detectMultiScale(gray_frame, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    for (x, y, w, h) in objects:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 0, 0), 2)  # Dibujar rectángulo en los objetos detectados
    return frame
