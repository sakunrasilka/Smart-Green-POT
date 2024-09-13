from ultralytics import YOLO
import cv2
import cvzone
import math


# Open the webcam
cap = cv2.VideoCapture(0)
cap.set(3, 1200)  # Set width
cap.set(4, 720)  # Set height


# Load the trained YOLO model
model = YOLO('best1.pt')

# Class names for the model (add more classes if needed)
classNames = ["plant_leaf"]

while True:
    success, img = cap.read()  # Capture frame from webcam
    results = model(img, stream=True)  # Get predictions from the model

    for r in results:
        boxes = r.boxes  # Get all detected boxes

        for box in boxes:
            # Get the bounding box coordinates
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            w, h = x2 - x1, y2 - y1


            # Get the confidence level of the prediction
            conf = math.ceil((box.conf[0] * 100)) / 100
            if conf>0.8:
                # Display the bounding box only if confidence is greater than 90%
                # Draw the bounding box using cvzone
                cvzone.cornerRect(img, (x1, y1, w, h), l=30, rt=5, colorR=(0, 255, 0))

                # Get the class index and display the class name and confidence
                cls = int(box.cls[0])
                cvzone.putTextRect(img, f'{classNames[cls]} {conf}', (max(0, x1), max(35, y1)), scale=1, thickness=1, colorR=(0, 255, 0))

    # Show the image with bounding boxes and class names
    cv2.imshow("Image", img)

    # Wait for 1ms, break if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and close windows
cap.release()
cv2.destroyAllWindows()
