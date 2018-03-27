#Requirements 

you will need python3 with tensorflow,numpy ,scipy, matplotib 

#Datasets 

load mnist image datasets from tensorflow by tensorflow.examples.tutorials.mnist

#Note 

GANs is very demanding for super parameters, and it also requires training a lot of wheels. In order for this job to be completed without GPU, we will do it on the MNIST data set, with 60000 of them as training sets and 10,000 as test sets. Each picture has a white number on a black background (0-9).

To simplify our code, we'll use the encapsulation of the TensorFlow MNIST, which downloads and loads the MNIST dataset, see this document. The default parameter will use 5000 training samples as the validation set. The data will be stored in a folder called MNIST_data. (Note:you can put it in the corresponding folder after you download the dataset.)
