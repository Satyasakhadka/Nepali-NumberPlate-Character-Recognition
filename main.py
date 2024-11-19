
from fastapi import FastAPI, File, UploadFile, Form, Request
from fastapi.responses import HTMLResponse, FileResponse
from fastapi.templating import Jinja2Templates
from pathlib import Path
import shutil
import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from ultralytics import YOLO
from fastapi.staticfiles import StaticFiles

# Create app
app = FastAPI()

# Serving static files (for the CSS)
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load models
model = YOLO('/Users/pc/Desktop/Nepali_license_plate/defense_best_model_final.pt')
cnn_model = load_model("/Users/pc/Desktop/Nepali_license_plate/best_model.keras")
license_plate_model = YOLO('/Users/pc/Desktop/Nepali_license_plate/license_plate_detector.pt')  # Update with the correct path in Colab


# Define image dimensions for model input
img_height, img_width = 32, 32

# Create a label mapping from class indices to characters
class_indices = {'क': 0, 'को': 1, 'ख': 2, 'ग': 3, 'च': 4, 'ज': 5, 'झ': 6, 'ञ': 7, 'डि': 8, 'त': 9, 'ना': 10, 'प': 11, 'प्र': 12, 'ब': 13, 'बा': 14, 'भे': 15, 'म': 16, 'मे': 17, 'य': 18, 'लु': 19, 'सी': 20, 'सु': 21, 'से': 22, 'ह': 23, '०': 24, '१': 25, '२': 26, '३': 27, '४': 28, '५': 29, '६': 30, '७': 31, '८': 32, '९': 33}
decoded = {v: k for k, v in class_indices.items()}

# Image preprocessing for prediction
def preprocess_image(img_array):
    img_resized = cv2.resize(img_array, (img_height, img_width))
    img_array = img_resized / 255.0
    return np.expand_dims(img_array, axis=0)

def recognize_character(cropped_char):
    img = preprocess_image(cropped_char)
    prediction = cnn_model.predict(img)
    predicted_class = np.argmax(prediction, axis=1)[0]
    return decoded[predicted_class]

# Sort bounding boxes by y-center for line grouping
def sort_boxes(boxes):
    if not boxes:
        print("No character bounding boxes detected!")
        return []

    heights = [abs(y2 - y1) for x1, y1, x2, y2 in boxes]
    avg_height = sum(heights) / len(heights)
    line_threshold = avg_height / 2

    boxes_with_center = []
    for x1, y1, x2, y2 in boxes:
        y_center = (y1 + y2) / 2
        boxes_with_center.append({'box': (x1, y1, x2, y2), 'y_center': y_center})

    boxes_with_center.sort(key=lambda b: b['y_center'])

    lines = []
    current_line = []
    current_y = None
    for b in boxes_with_center:
        y_center = b['y_center']
        if current_y is None or abs(y_center - current_y) <= line_threshold:
            current_line.append(b)
            current_y = y_center
        else:
            lines.append(current_line)
            current_line = [b]
            current_y = y_center
    if current_line:
        lines.append(current_line)

    for line in lines:
        line.sort(key=lambda b: b['box'][0])

    sorted_boxes = [b['box'] for line in lines for b in line]
    return sorted_boxes

# Main function to process the number plate image
def process_car_image(image_path):
    image = cv2.imread(image_path)

    # Step 1: Detect license plate
    license_plate_results = license_plate_model(image)
    license_boxes = []
    for detection in license_plate_results[0].boxes:
        license_boxes.append([int(coord) for coord in detection.xyxy[0]])
    
    if not license_boxes:
        print("No license plate detected!")
        return None

    x1, y1, x2, y2 = license_boxes[0]
    cropped_license_plate = image[y1:y2, x1:x2]

    # Step 2: Detect characters on the license plate
    character_results = model(cropped_license_plate)
    character_boxes = []
    for detection in character_results[0].boxes:
        character_boxes.append([int(coord) for coord in detection.xyxy[0]])

    if not character_boxes:
        print("No characters detected on the license plate!")
        return None

    sorted_character_boxes = sort_boxes(character_boxes)

    detected_characters = []
    for x1, y1, x2, y2 in sorted_character_boxes:
        cropped_char = cropped_license_plate[y1:y2, x1:x2]
        character = recognize_character(cropped_char)
        detected_characters.append(character)

    detected_text = ''.join(detected_characters)

    # Draw bounding boxes on the image
    for x1, y1, x2, y2 in sorted_character_boxes:
        cv2.rectangle(cropped_license_plate, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Save the image with bounding boxes
    output_image_path = "static/output_image.jpg"
    cv2.imwrite(output_image_path, image)

    return detected_text, output_image_path

# Templates initialization
templates = Jinja2Templates(directory="templates")

# Route to upload the image and display result
@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "detected_text": None, "image_path": None})

@app.post("/upload", response_class=HTMLResponse)
async def upload_image(request: Request, image: UploadFile = File(...)):
    # Save uploaded image
    image_path = f"uploads/{image.filename}"
    with open(image_path, "wb") as f:
        shutil.copyfileobj(image.file, f)

    # Process the image
    detected_text, image_with_boxes_path = process_car_image(image_path)

    return templates.TemplateResponse("index.html", {"request": request, "detected_text": detected_text, "image_path": image_with_boxes_path})


