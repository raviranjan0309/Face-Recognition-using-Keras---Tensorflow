The whole process for face recognition using Keras can be divided in four major steps:</br>
a.	Detect/ Identify faces in an image using Dlib and opencv</br>
b.	Convert image into grayscale and crop into 200X200 pixels</br>
c.	Design convolutional neural network using Keras</br>
d.	Train the model on the test model on testing data</br></br>
Convolutional Neural Networks (ConvNets or CNNs) are a category of Neural Networks that have proven very effective in areas such as image recognition and classification.</br></br>
There are four main operations in the ConvNet:</br>
•	Convolution</br>
•	Non Linearity (ReLU)</br>
•	Pooling or Sub Sampling</br>
•	Classification (Fully Connected Layer)</br></br>
These operations are the basic building blocks of every Convolutional Neural Network, so understanding how these work is an important step to developing a sound understanding of ConvNets.</br></br>
An Image is a matrix of pixel values. Essentially, every image can be represented as a matrix of pixel values.</br></br>

 
Channel is a conventional term used to refer to a certain component of an image. An image from a standard digital camera will have three channels – red, green and blue – one can imagine those as three 2d-matrices stacked over each other (one for each color), each having pixel values in the range 0 to 255.</br></br>

A grayscale image, on the other hand, has just one channel. We will only consider grayscale images, so we will have a single 2d matrix representing an image. The value of each pixel in the matrix will range from 0 to 255 – zero indicating black and 255 indicating white.
ConvNets derive their name from the “convolution” operator. The primary purpose of Convolution in case of a ConvNet is to extract features from the input image. Convolution preserves the spatial relationship between pixels by learning image features using small squares of input data. We will not go into the mathematical details of Convolution here, but will try to understand how it works over images.</br></br>
I will use the VGG-Face model as an example. Briefly, the VGG-Face model is the same NeuralNet architecture as the VGG16 model used to identity 1000 classes of object in the ImageNet competition. The VGG16 name simply states the model originated from the Visual Geometry Group and that it was 16 trainable layers. The main difference between the VGG16-ImageNet and VGG-Face model is the set of calibrated weights as the training sets were different.</br></br>
The model architecture is a linear sequence of layer transformations of the following types:</br>
•	Convolution + ReLU activations</br>
•	MaxPooling</br>
•	Softmax</br>
Keras is a high-level library, used specially for building neural network models. It is written in Python and is compatible with both Python – 2.7 & 3.5. Keras was specifically developed for fast execution of ideas. It has a simple and highly modular interface, which makes it easier to create even complex neural network models. This library abstracts low level libraries, namely Theano and TensorFlow so that, the user is free from “implementation details” of these libraries.</br></br>
Technical Specification:</br>
•	Python 3.5</br>
•	Packages – numpy, scipy, dlib, OpenCV, Keras, Tensorflow</br>
•	Set backend as Tensorflow in json file of Keras</br>
•	Dataset: create your own dataset</br>

