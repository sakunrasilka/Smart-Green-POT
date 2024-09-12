# Smart-Green-POT

Train_Plant_Disease.ipynb --> This is the CNN layer I have created for the plant disease detection Model.
  Here I have used Tenserflow, Matplotlib, Pandas and Seaborn. I downloaded a train set from kaggle with 70295 files in 38 classes. These 38 classes are different types of diseased and healthy plants leaves of Tomatoe,Apple,Cherry,Corn,Grape,Strawberry and Pepper. ex:- Tomato_healthy, Tomato_Bacterial_spot, Grape_Leaf_blight...

For the Validation set i used another data set from kaggle with 17572 files belon to the 38 classes. To avoid overshooting, I used a smaller learning rate of 0.0001, since there is a chance of underfitting, i increase the number of neurons and the layers of convolution.

Used conv2D layers by changing the number of filters to create the CNN layer. This layer learns texture,edges and patterns from the input image and outputs a 3D tensor of (128,128,no_of_filters)


