# import cv2
# import numpy as np
# from tensorflow.keras.models import load_model
# from ultralytics import YOLO
# import matplotlib.pyplot as plt

# # Load YOLO model and your CNN model
# model = YOLO('/Users/pc/Desktop/Nepali_license_plate/defense_best_model_final.pt')  # Adjust path if needed
# cnn_model = load_model("/Users/pc/Desktop/Nepali_license_plate/best_model.keras")

# # Define image dimensions for model input
# img_height, img_width = 32, 32

# # Create a label mapping from class indices to characters
# class_indices = {'क': 0, 'को': 1, 'ख': 2, 'ग': 3, 'च': 4, 'ज': 5, 'झ': 6, 'ञ': 7, 'डि': 8, 'त': 9, 'ना': 10, 'प': 11, 'प्र': 12, 'ब': 13, 'बा': 14, 'भे': 15, 'म': 16, 'मे': 17, 'य': 18, 'लु': 19, 'सी': 20, 'सु': 21, 'से': 22, 'ह': 23, '०': 24, '१': 25, '२': 26, '३': 27, '४': 28, '५': 29, '६': 30, '७': 31, '८': 32, '९': 33}
# decoded = {v: k for k, v in class_indices.items()}

# # Image preprocessing for prediction
# def preprocess_image(img_array):
#     img_resized = cv2.resize(img_array, (img_height, img_width))
#     img_array = img_resized / 255.0
#     return np.expand_dims(img_array, axis=0)

# def recognize_character(cropped_char):
#     img = preprocess_image(cropped_char)
#     prediction = cnn_model.predict(img)
#     predicted_class = np.argmax(prediction, axis=1)[0]
#     return decoded[predicted_class]

# # Sort bounding boxes by y-center for line grouping
# def sort_boxes(boxes):
#     heights = [abs(y2 - y1) for x1, y1, x2, y2 in boxes]
#     avg_height = sum(heights) / len(heights)
#     line_threshold = avg_height / 2

#     boxes_with_center = []
#     for x1, y1, x2, y2 in boxes:
#         y_center = (y1 + y2) / 2
#         boxes_with_center.append({'box': (x1, y1, x2, y2), 'y_center': y_center})

#     boxes_with_center.sort(key=lambda b: b['y_center'])

#     lines = []
#     current_line = []
#     current_y = None
#     for b in boxes_with_center:
#         y_center = b['y_center']
#         if current_y is None or abs(y_center - current_y) <= line_threshold:
#             current_line.append(b)
#             current_y = y_center
#         else:
#             lines.append(current_line)
#             current_line = [b]
#             current_y = y_center
#     if current_line:
#         lines.append(current_line)

#     for line in lines:
#         line.sort(key=lambda b: b['box'][0])

#     sorted_boxes = [b['box'] for line in lines for b in line]
#     return sorted_boxes

# # Main function to process the number plate image
# def process_number_plate(image_path):
#     # Load image
#     image = cv2.imread(image_path)
#     results = model(image)

#     # Collect the bounding boxes
#     boxes = []
#     for detection in results[0].boxes:
#         x1, y1, x2, y2 = map(int, detection.xyxy[0])
#         boxes.append((x1, y1, x2, y2))

#     # Sort the boxes
#     sorted_boxes = sort_boxes(boxes)

#     detected_characters = []

#     # Loop through each sorted box and recognize characters
#     for x1, y1, x2, y2 in sorted_boxes:
#         # Crop detected character from image
#         cropped_char = image[y1:y2, x1:x2]

#         # Recognize character using CNN
#         character = recognize_character(cropped_char)
#         detected_characters.append(character)

#         # Optionally, draw a box and label the detected character
#         cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Draw bounding box

#     # Prepare the text to display on the image
#     detected_text = ''.join(detected_characters)
#     #cv2.putText(image, f"Detected License Plate: {detected_text}", (10, image.shape[0] - 20),
#                # cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

#     # Display the image with detected characters
#     plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
#     plt.axis('off')
#     plt.show()

#     # Print the detected text in the console
#     print("Detected License Plate Characters:", detected_text)

# # Run the pipeline on an example image
# image_path = "/Users/pc/Desktop/Nepali_license_plate/2021-03-25_14_32_11.jpg"  # Replace with the actual path

# process_number_plate(image_path)




# import os
# import shutil
# from fastapi import FastAPI, File, UploadFile
# from fastapi.responses import HTMLResponse
# from fastapi.staticfiles import StaticFiles
# from tensorflow.keras.models import load_model
# import cv2
# import numpy as np
# from ultralytics import YOLO
# import matplotlib.pyplot as plt
# from io import BytesIO
# from PIL import Image

# app = FastAPI()

# # Load YOLO model and your CNN model
# model = YOLO('/Users/pc/Desktop/Nepali_license_plate/defense_best_model_final.pt')
# cnn_model = load_model("/Users/pc/Desktop/Nepali_license_plate/best_model.keras")

# # Define image dimensions for model input
# img_height, img_width = 32, 32

# # Create a label mapping from class indices to characters
# class_indices = {'क': 0, 'को': 1, 'ख': 2, 'ग': 3, 'च': 4, 'ज': 5, 'झ': 6, 'ञ': 7, 'डि': 8, 'त': 9, 'ना': 10, 'प': 11, 'प्र': 12, 'ब': 13, 'बा': 14, 'भे': 15, 'म': 16, 'मे': 17, 'य': 18, 'लु': 19, 'सी': 20, 'सु': 21, 'से': 22, 'ह': 23, '०': 24, '१': 25, '२': 26, '३': 27, '४': 28, '५': 29, '६': 30, '७': 31, '८': 32, '९': 33}
# decoded = {v: k for k, v in class_indices.items()}

# # Image preprocessing for prediction
# def preprocess_image(img_array):
#     img_resized = cv2.resize(img_array, (img_height, img_width))
#     img_array = img_resized / 255.0
#     return np.expand_dims(img_array, axis=0)

# def recognize_character(cropped_char):
#     img = preprocess_image(cropped_char)
#     prediction = cnn_model.predict(img)
#     predicted_class = np.argmax(prediction, axis=1)[0]
#     return decoded[predicted_class]

# # Sort bounding boxes by y-center for line grouping
# def sort_boxes(boxes):
#     heights = [abs(y2 - y1) for x1, y1, x2, y2 in boxes]
#     avg_height = sum(heights) / len(heights)
#     line_threshold = avg_height / 2

#     boxes_with_center = []
#     for x1, y1, x2, y2 in boxes:
#         y_center = (y1 + y2) / 2
#         boxes_with_center.append({'box': (x1, y1, x2, y2), 'y_center': y_center})

#     boxes_with_center.sort(key=lambda b: b['y_center'])

#     lines = []
#     current_line = []
#     current_y = None
#     for b in boxes_with_center:
#         y_center = b['y_center']
#         if current_y is None or abs(y_center - current_y) <= line_threshold:
#             current_line.append(b)
#             current_y = y_center
#         else:
#             lines.append(current_line)
#             current_line = [b]
#             current_y = y_center
#     if current_line:
#         lines.append(current_line)

#     for line in lines:
#         line.sort(key=lambda b: b['box'][0])

#     sorted_boxes = [b['box'] for line in lines for b in line]
#     return sorted_boxes

# # Main function to process the number plate image
# def process_number_plate(image_path):
#     image = cv2.imread(image_path)
#     results = model(image)

#     # Collect the bounding boxes
#     boxes = []
#     for detection in results[0].boxes:
#         x1, y1, x2, y2 = map(int, detection.xyxy[0])
#         boxes.append((x1, y1, x2, y2))

#     sorted_boxes = sort_boxes(boxes)

#     detected_characters = []

#     # Loop through each sorted box and recognize characters
#     for x1, y1, x2, y2 in sorted_boxes:
#         cropped_char = image[y1:y2, x1:x2]
#         character = recognize_character(cropped_char)
#         detected_characters.append(character)

#     detected_text = ''.join(detected_characters)
#     return detected_text

# # Create upload folder if not exists
# UPLOAD_FOLDER = "uploaded_images"
# os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# @app.get("/", response_class=HTMLResponse)
# async def main():
#     content = """
#     <html>
#         <body>
#             <h2>Upload License Plate Image</h2>
#             <form action="/upload/" enctype="multipart/form-data" method="post">
#                 <input type="file" name="file">
#                 <input type="submit">
#             </form>
#         </body>
#     </html>
#     """
#     return content

# @app.post("/upload/")
# async def upload_file(file: UploadFile = File(...)):
#     file_location = os.path.join(UPLOAD_FOLDER, file.filename)
#     with open(file_location, "wb") as f:
#         shutil.copyfileobj(file.file, f)

#     # Process the uploaded image
#     detected_text = process_number_plate(file_location)

#     # Return the detected text as the response
#     return HTMLResponse(content=f"""
#     <html>
#         <body>
#             <h2>Detected License Plate Characters:</h2>
#             <p>{detected_text}</p>
#             <img src="/static/{file.filename}" alt="Uploaded Image">
#         </body>
#     </html>
#     """)

# # Serve uploaded images statically
# app.mount("/static", StaticFiles(directory=UPLOAD_FOLDER), name="static")


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
def process_number_plate(image_path):
    image = cv2.imread(image_path)
    results = model(image)

    boxes = []
    for detection in results[0].boxes:
        x1, y1, x2, y2 = map(int, detection.xyxy[0])
        boxes.append((x1, y1, x2, y2))

    sorted_boxes = sort_boxes(boxes)

    detected_characters = []
    for x1, y1, x2, y2 in sorted_boxes:
        cropped_char = image[y1:y2, x1:x2]
        character = recognize_character(cropped_char)
        detected_characters.append(character)

    detected_text = ''.join(detected_characters)

    # Draw bounding boxes on the image
    for x1, y1, x2, y2 in sorted_boxes:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)

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
    detected_text, image_with_boxes_path = process_number_plate(image_path)

    return templates.TemplateResponse("index.html", {"request": request, "detected_text": detected_text, "image_path": image_with_boxes_path})


