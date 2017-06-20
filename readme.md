#**Traffic Sign Recognition** 

-------

**Design a Convolutional Neural Network to Classify Traffic Signs**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data samples
* Design, train and test the model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images


[//]: # (Image References)

[dataset-visualization]: ./dataset-distribution.png "Dataset Visualization"
[before-after-preprocessing]: ./before-after-preprocessing.png "Before and After Preprocessing"
[sign1]: ./new-images/sign-1.jpg "30km/h"
[sign2]: ./new-images/sign-7.jpg "100km/h"
[sign3]: ./new-images/sign-11.jpg "Right of way in next intersection"
[sign4]: ./new-images/sign-13.jpg "Yield"
[sign5]: ./new-images/sign-17.jpg "No Entry"
[sign6]: ./new-images/sign-18.jpg "General Caution"
[sign7]: ./new-images/sign-22.jpg "Bumpy Road"
[sign8]: ./new-images/sign-25.jpg "Road Work Ahead"
[sign9]: ./new-images/sign-26.jpg "Road Work Ahead"
[sign10]: ./new-images/signchallenge-1.jpg "Road Work Ahead"
[model-softmax]: ./model-softmax.png "Softmax"
[conv1]: ./conv1.png "Softmax"
[conv2]: ./conv2.png "Softmax"
[avg1]: ./avg1.png "Softmax"
[avg2]: ./avg2.png "Softmax"
[avg3]: ./avg3.png "Softmax"

---

### Data Set Summary & Exploration

#### 1. Training, Validation and Testing Data Size

I used numpy to calculate summary statistics of the traffic signs data set:

* Size of training set: **34799**
* Size of the validation set: **4410**
* Size of test set: **12630**
* Shape of a traffic sign image: **32x32 (RGB)**
* The number of unique classes/labels: **43**


#### 2. Distribution of Training Data

Here is an exploratory visualization of the data set.
The bar chart shows the distribution of training samples by class label.

![alt text][dataset-visualization]

### Convolutional Neural Network (CNN) Model Architecture

#### 1. Preprocessing Data Sets

First step was grayscaling the images to increase performance during training.
However, there may be some cases where the color can be significant, although I
haven't explored this.

The image is also normalized to improve training performance.

Here is an example of a traffic sign image before and after grayscaling and normalization.

![alt text][before-after-preprocessing]


#### 2. Model Architecture

My final model consisted of the following layers. I started with the LeNet architecture and
then followed the paper on GoogLeNet by adding in inception modules, average pooling
followed by 2 fully connected layers with dropout in between both.


| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x3 RGB image   							|
| Preprocessing         | 32x32x1 Grayscale normalized image            |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 28x28x6 	|
| RELU 1				| Activation                                    |
| Convolution 5x5     	| 1x1 stride, valid padding, outputs 24x24x16 	|
| RELU 2				| Activation									|
| Average pooling 1	    | 2x2 stride, outputs 12x12x16 				    |
| Inception Module 1	| outputs 12x12x256 				            |
| Average pooling 2	    | 2x2 stride, outputs 6x6x256 				    |
| Inception Module 2	| outputs 6x6x512    				            |
| Average pooling 3	    | 2x2 stride, same padding, outputs 6x6x512 	|
| Dropout 1	            | Flatten the output and dropout 50% layer      |
| Fully connected 1		| Outputs 9216x512        						|
| Dropout 2	            | Dropout 50% layer                             |
| Fully connected 2		| Outputs 512x43        						|
| Softmax				| Softmax probabilities        				    |
 

#### 3. Training the Model

- Preprocessed the images
- One hot encoded the labels
- Trained using Adam optimizer
- Batch size: 128 images
- Learning rate: 0.001
- Variables initialize with truncated normal distributions
- Hyperparameters, mu = 0, sigma = 0.1, dropout = 0.5
- Trained for 20 epochs

Training was carried out on an AWS g2.2xlarge GPU instance. Each epoch was completed
in around 24s, total training time of about 6-8 minutes.

#### 4. Designing the Model 

Final model results:
- Validation set accuracy: **0.97** 
- Test set accuracy: **0.95**

The approach started with a review of LeNet and GoogLeNet architectures and those were
used as a starting point to decide how to order and how many inception modules, pooling
convolution and dropout layers to use.

* First started with LeNet and with minor tweaks, achieved an accuracy of about 0.89
* Reduced and switched max pooling to average pooling to get to 0.90
* Added an inception module after the first two convolutions to get to 0.90-0.93
* At this point, there were very deep inception modules (referencing GoogLeNet) and 4 fully connected layers and the model appeared to be overfitting.
* Reducing the number of fully connected layers to 2 and adding dropout between them made the largest impact and achieved around 0.97 accuracy. 

Once I noticed the accuracy starting high and surpassing the minimium 0.93, I tweaked the layers by trial and error but my approach was to try and
keep as much information flowing through the model while balancing the dimensional reduction. 
This led me to have 2 convolution layers with small patch sizes, average pooling with small patch size and do more sever reductions with the inception modules.


### Test a Model on New Images

#### 1. New Test Images

Here are five German traffic signs that I found on the web:

![alt text][sign1]
![alt text][sign2]
![alt text][sign3]
![alt text][sign4]
![alt text][sign5]
![alt text][sign6]
![alt text][sign7]
![alt text][sign8]
![alt text][sign9]
![alt text][sign10]

The signs are generally easy to identify, I chose a few that are at angles and one that had a "1" spray painted
onto a 30km/h sign. 

#### 2. New Image Test Results

Here are the results of the prediction:

| Image			        |     Prediction	        					| 
|:---------------------:|:---------------------------------------------:| 
| Yield           		| Yield                                         | 
| Traffic Signals       | Traffic Signals             				    |
| General Caution       | General Caution                               |
| Speed Limit 30        | Speed Limit 30                                |
| Right of Way          | Right of Way                                  |
| Road Work             | Road Work                                     |
| Bumpy Road            | Bicycles Crossing                             |
| No Entry              | No Entry                                      |
| Speed Limit 100       | Roundabout Mandatory                          |
| Speed Limit 30        | Speed Limit 30                                |


The model was able to correctly guess 8 of the 10 traffic signs, which gives an accuracy of **80%**. 
I was impressed that the spray painted sign still classified correctly event though the clear 100km/h sign was wrong.

The bicycles crossing and bumpy road signs do look very similar at low resolution, so I can understand that wrong prediction. 

#### 3. Model Certainty

The following shows the top 5 predictions for each sign.

![alt text][model-softmax]


#### 4. Visualizing the Model Layers

**First Convolution**

![alt text][conv1]

**Second Convolution**

![alt text][conv2]

**First Avg Pool**

![alt text][avg1]

**Second Avg Pool**

![alt text][avg2]

**Third Avg Pool**

![alt text][avg3]
