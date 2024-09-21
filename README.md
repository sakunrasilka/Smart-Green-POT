# Smart-Green-POT

### Train_Plant_Disease.ipynb -->
This is the CNN layer I have created for the plant disease detection Model. Here I have used Tenserflow, Matplotlib, Pandas and Seaborn. I downloaded a train set from kaggle with 70295 files in 38 classes. These 38 classes are different types of diseased and healthy plants leaves of Tomatoe,Apple,Cherry,Corn,Grape,Strawberry and Pepper. ex:- Tomato_healthy, Tomato_Bacterial_spot, Grape_Leaf_blight...

For the Validation set i used another data set from kaggle with 17572 files belon to the 38 classes. To avoid overshooting, I used a smaller learning rate of 0.0001, since there is a chance of underfitting, i increase the number of neurons and the layers of convolution.

Used Conv2D layers by changing the number of filters to create the CNN layer. This layer learns texture,edges and patterns from the input image and outputs a 3D tensor of (128,128,no_of_filters). In the code i used two Conv2D layers and a MaPool2D layer which select the strongest neuron activation and prevent overfitting.

Given below is the CNN layer which I have created for the model.

<img src="https://github.com/user-attachments/assets/49ccb790-8ddd-4867-8536-f7af5097753e" alt="CNN Layer" width="400"/>

Given Below is the accuracy visualization of my model.

<img src="https://github.com/user-attachments/assets/57eb61e5-f970-4a38-b49f-6139ad054a2d" alt="CNN Layer" width="400"/>

### Webcam.py -->
I used the object detection to identify the plant leaves in real time by using opencv and yolov8
I have created a custom trained object detection model for tomatoe leaves. It identify the tomatoe leaves and makes boundary boxes around the exact leaf
pip install -U ultralytics

Image uploaded

<img src="https://github.com/user-attachments/assets/c095f14d-55b3-439c-ac73-a82c16f81aac" alt="CNN Layer" width="400"/>

Real time Webcam

<img src="https://github.com/user-attachments/assets/9a9eb9de-6d8c-4923-9537-22300bebb5c9" alt="CNN Layer" width="400"/>

### YoloTrain.ipynb -->
I used google colab and its gpu to get the weights for a pretrained tomatoe leaf detection model.
can send the train,valid and test files to yolov8 and get the  weights for the model.








