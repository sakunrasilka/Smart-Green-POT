# Smart-Green-POT

Train_Plant_Disease.ipynb --> This is the CNN layer I have created for the plant disease detection Model.
  Here I have used Tenserflow, Matplotlib, Pandas and Seaborn. I downloaded a train set from kaggle with 70295 files in 38 classes. These 38 classes are different types of diseased and healthy plants leaves of Tomatoe,Apple,Cherry,Corn,Grape,Strawberry and Pepper. ex:- Tomato_healthy, Tomato_Bacterial_spot, Grape_Leaf_blight...

For the Validation set i used another data set from kaggle with 17572 files belon to the 38 classes. To avoid overshooting, I used a smaller learning rate of 0.0001, since there is a chance of underfitting, i increase the number of neurons and the layers of convolution.

Used Conv2D layers by changing the number of filters to create the CNN layer. This layer learns texture,edges and patterns from the input image and outputs a 3D tensor of (128,128,no_of_filters). In the code i used two Conv2D layers and a MaPool2D layer which select the strongest neuron activation and prevent overfitting.
![SmartSelect_20240912_084633_Samsung Notes](https://github.com/user-attachments/assets/49ccb790-8ddd-4867-8536-f7af5097753e)



