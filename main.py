"""Main file for medicine detection"""

import os
import re
import sys
from typing import Optional

import cv2
import numpy as np
from ultralytics import YOLO
from voice_decoder.voice_decoder import VoiceDecoder

from ocr_pipeline import OCRPipeline

# Paths
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_DIR = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_DIR)

# Initialize models
ocr = OCRPipeline()
model = YOLO("models/best.pt")
decoder = VoiceDecoder()


def preprocess_image(img: np.ndarray) -> np.ndarray:
    """
    Preprocess the image for better OCR performance.

    parameters:
        img (np.ndarray): RGB input image.

    Returns:
        np.ndarray: Thresholded and sharpened grayscale image.
    """
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    blur = cv2.GaussianBlur(gray, (3, 3), 0)
    sharpen_kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(blur, -1, sharpen_kernel)
    _, thresh = cv2.threshold(sharpened, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return thresh


def clean_text(text: str) -> str:
    """
    Clean OCR output by filtering invalid or short words.

    parameters:
        text (str): Raw text extracted by OCR.

    Returns:
        str: Cleaned and filtered text.
    """
    words = re.findall(r"\b[a-zA-Záéíóúãõâêôç]{2,}\b", text.lower())
    clean_words = [w for w in words if len(w) > 3]
    return " ".join(clean_words)


def process_ocr(crop: np.ndarray) -> str:
    """
    Run OCR pipeline on a cropped image.

    parameters:
        crop (np.ndarray): Cropped BGR image from frame.

    Returns:
        str: Cleaned text extracted from the crop.
    """
    cropped_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
    processed_img = preprocess_image(cropped_rgb)
    ocr.image_to_string(processed_img)
    text = ocr.processed_text_output
    cleaned_text = clean_text(text)
    return cleaned_text


def is_stable(
    last_bbox: Optional[np.ndarray], current_bbox: np.ndarray, threshold: int = 10
) -> bool:
    """
    Check if the detection bounding box is stable between frames.

    parameters:
        last_bbox (Optional[np.ndarray]): Previous bounding box coordinates.
        current_bbox (np.ndarray): Current bounding box coordinates.
        threshold (int): Maximum allowed movement to be considered stable.

    Returns:
        bool: True if movement is below threshold, False otherwise.
    """
    if last_bbox is None:
        return False
    movement = np.linalg.norm(current_bbox - last_bbox)
    return movement < threshold


def detection_pipeline() -> None:
    """
    Main detection and OCR pipeline.
    Captures video, detects medicine packages, waits for stability, runs OCR, and reads text aloud.
    """
    cap = cv2.VideoCapture(0)

    last_bbox: Optional[np.ndarray] = None
    stable_counter: int = 0
    stable_required: int = 5  # Frames required to be stable before OCR

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        results = model(frame)[0]
        annotated_frame = results.plot()
        cv2.imshow("YOLO Detection", annotated_frame)

        if len(results.boxes) > 0:
            x1, y1, x2, y2 = map(int, results.boxes[0].xyxy[0])
            current_bbox = np.array([x1, y1, x2, y2])

            if is_stable(last_bbox, current_bbox):
                stable_counter += 1
            else:
                stable_counter = 0

            last_bbox = current_bbox

            if stable_counter >= stable_required:
                crop = frame[y1:y2, x1:x2]
                text = process_ocr(crop)

                if text.strip():
                    print(f"OCR output: {text}")
                    try:
                        decoder.string_to_speech(text)
                    except Exception as e:
                        print(f"Decoder error: {e}")

                stable_counter = 0  # Reset counter after successful OCR

        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    detection_pipeline()
