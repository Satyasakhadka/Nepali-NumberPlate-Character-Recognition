# Nepali Number Plate Character Recognition Project

This project is developed as part of the **Advanced Data Science** elective course offered by Samsung. The goal of this project is to recognize Nepali number plates characters from vehicle images by utilizing deep learning techniques, specifically YOLO and CNN models.

## Contributors:
- [Satyasa Khadka](https://github.com/Satyasakhadka)
- [Sandhya Baral](https://github.com/Sandukkk)
- [Sudip Tiwari](https://github.com/sudiptiwari)
---

## Overview

The system is designed to perform the following tasks:  
1. **Number Plate Detection:**  
   - A **pretrained YOLO model** is used to detect the number plate from an image of a vehicle.  
   
2. **Character Detection in Number Plate:**  
   - A second **YOLO model**, trained on the **Inspiring Lab dataset**, detects individual characters in the number plate. This dataset contains images of Nepali license plates, making the model well-suited for this task.  

3. **Character Recognition:**  
   - Detected characters are cropped from the number plate image and fed into a **custom-built CNN model**.  
   - This CNN model, trained on the **Inspiring Lab dataset** (which includes characters frequently used in Nepali number plates), identifies the characters.  

---

## Key Features

- **Pretrained YOLO Model:** Efficiently detects number plates from vehicle images.  
- **Custom YOLO Model:** Specifically trained to detect characters on Nepali license plates.  
- **Custom CNN Model:** Designed for character recognition using data tailored to the Nepali context.  
- **Dataset:** Utilizes the **Inspiring Lab dataset**, ensuring high accuracy for the unique structure of Nepali license plates.  

---

## Methodology

1. **Number Plate Detection:**  
   - The pretrained YOLO model takes an image of a vehicle and identifies the region containing the number plate.

2. **Character Detection:**  
   - The custom YOLO model identifies individual characters in the detected number plate region.  
   - Detected characters are cropped for further processing.

3. **Character Recognition:**  
   - Cropped characters are passed to the custom-built CNN model, which outputs the recognized character.  
   - The CNN model is trained on a dataset of characters commonly used in Nepali number plates to ensure high recognition accuracy.

---

## Technologies Used

- **YOLO (You Only Look Once):**  
   - Pretrained model for number plate detection.  
   - Custom-trained model for character detection.  
   
- **Convolutional Neural Network (CNN):**  
   - Custom model for character recognition.  

- **Datasets:**  
   - **YOLO Training Dataset:** [Nepali Vehicle Number Plate Dataset](https://www.kaggle.com/datasets/inspiring-lab/nepali-vehicles-number-plate-dataset)  
   - **CNN Training Dataset:** [Nepali Number Plate Characters Dataset](https://www.kaggle.com/datasets/inspiring-lab/nepali-number-plate-characters-dataset/discussion?sort=hotness)  

---

## Applications

- Automated vehicle registration systems.  
- Traffic monitoring and management.  
- Smart parking systems.  

---

This project demonstrates the effective application of deep learning models to solve a real-world problem, leveraging pretrained and custom-trained networks for detection and recognition tasks.
## Demo Video

🎥 **Watch the demo video of our project:** 

https://github.com/user-attachments/assets/fc7080cb-cff8-4f0a-98ea-619cd41fd854

## Example of Embossed Plate detection

![Embossed Plate detection](Screenshot 2024-11-22 091150.png)




