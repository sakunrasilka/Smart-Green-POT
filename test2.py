from ultralytics import YOLO
import cv2
import cvzone
import math
import tensorflow as tf
import numpy as np

# Load the trained YOLO model
model = YOLO('best1.pt')

# Load the trained CNN model
cnn = tf.keras.models.load_model('trained_model.keras')

# Class names for the model
classNames = ["plant_leaf"]

# Path to the input image
image_path = "download.jpg"  # Replace with your image file path
img = cv2.imread(image_path)

# Perform object detection on the image
results = model(img, show=False)

for r in results:
    boxes = r.boxes  # Get all detected boxes

    for box in boxes:
        # Get the bounding box coordinates
        x1, y1, x2, y2 = box.xyxy[0]
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        w, h = x2 - x1, y2 - y1

        # Get the confidence level of the prediction
        conf = math.ceil((box.conf[0] * 100)) / 100
        if conf > 0.9:  # Adjust confidence threshold as needed
            # Draw the bounding box
            cvzone.cornerRect(img, (x1, y1, w, h), l=30, rt=5, colorR=(0, 255, 0))

            # Extract the ROI
            roi = img[y1:y2, x1:x2]
            roi_resized = cv2.resize(roi, (128, 128))  # Resize for CNN input
            roi_array = np.array([roi_resized])  # Create batch

            # Predict disease
            predictions = cnn.predict(roi_array)
            result_index = np.argmax(predictions)

            # Load validation set class names (if needed)
            validation_set = tf.keras.utils.image_dataset_from_directory(
                    'valid', labels="inferred", label_mode="categorical", image_size=(128, 128)
                )
            class_name = validation_set.class_names
            model_prediction = class_name[result_index]

            # Display prediction
            cvzone.putTextRect(img, f'Disease: {model_prediction}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=(0, 255, 0))



# Show the image with bounding boxes and predictions
cv2.imshow("Image", img)
cv2.waitKey(0)  # Wait until a key is pressed
cv2.destroyAllWindows()
