# **Behavioral Cloning** 

## Writeup

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/placeholder.png "Model Visualization"
[image2]: ./output/center_2019_02_04_03_12_00_332.jpg "Grayscaling"
[image3]: ./output/center_2019_02_11_07_34_51_598.jpg "Recovery Image"
[image4]: ./output/center_2019_02_11_07_34_53_697.jpg "Recovery Image"
[image5]: ./output/center_2019_02_11_07_34_58_248.jpg "Recovery Image"
[image6]: ./examples/placeholder_small.png "Normal Image"
[image7]: ./examples/placeholder_small.png "Flipped Image"
[image8]: ./output/data_with_dropout.png "Learning curve"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results
* ./model_1/output_video.mp4 for result video of driving the car in autonomous mode

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I used End-to-end driving model from NVidia's article.
My model consists of a convolution neural network with 5x5 filter sizes and depths between 24 and 48 (model.py lines 94-98) 
After that, I added convolution neural network with 3x3 filter sizes and depths 64 (model.py lines 100-102) 

The model includes RELU layers to introduce nonlinearity (code line 94-111), and the data is normalized in the model using a Keras lambda layer (code line 92). 

#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 95-112). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 120). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 120).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road, and driving unti-clockwise.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to create simple end-to-end model, then output learning curve, and added dropout layer to avoid overfitting.

My first step was to use a convolution neural network model similar to the NVidia's end-to-end driving. I thought this model might be appropriate because learning rate is good, and test result is good.

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I added dropout layer to the model to avoid overfitting.

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track. To improve the driving behavior in these cases, I added training data to be able to recover back to center.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 88-115) consisted of a convolution neural network and fully connected network.

<!-- Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric) -->

<!-- ![alt text][image1] -->

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded one laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to the way of recover to the center. These images show what a recovery looks like starting from :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

<!-- To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc .... -->

After the collection process, I had 10013 number of training data and 2504 validation data. I then preprocessed this data by normalize each pixels.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 10 as evidenced by learning curve below. I used an adam optimizer so that manually training the learning rate wasn't necessary.
![alt text][image8]
