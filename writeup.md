# **Behavioral Cloning** 

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report
---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* [model.py](https://github.com/stephenvfg/behavioral-cloning/blob/master/model.py) containing the script to create and train the model
* [drive.py](https://github.com/stephenvfg/behavioral-cloning/blob/master/drive.py) for driving the car in autonomous mode
* [model.h5](https://github.com/stephenvfg/behavioral-cloning/blob/master/model.h5) containing a trained convolution neural network 
* [writeup.md](https://github.com/stephenvfg/behavioral-cloning/blob/master/writeup.md) summarizing the results - this document!

#### 2. Submission includes functional code

Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The [model.py](https://github.com/stephenvfg/behavioral-cloning/blob/master/model.py) file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

I chose to to build my model based on the [NVIDIA deep learning model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) that was developed specifically for self-driving cars. Simply put, I trusted their team of researchers to develop a powerful model and architecture to solve this challenge better than I could.

The model consists of a convolutional neural network with 3x3 and 5x5 filter sizes all with RELU activations (model.py lines 70-80).

There is a Lambda pre-processing layer to normalize the data followed by a 2D cropping layer to remove unhelpful pixel data from the input images (model.py lines 65-66).

#### 2. Attempts to reduce overfitting in the model

The data that I gatehered for this model was split into two separated groups of training data and validation data (model.py lines 54-59). The model also contains a 25% dropout layer (model.py line 75) in order to reduce overfitting.




![Example of center driving on Track 1](https://github.com/stephenvfg/behavioral-cloning/blob/master/writeup-assets/track-1-middle-example.gif)

![Example of recovery driving on Track 1](https://github.com/stephenvfg/behavioral-cloning/blob/master/writeup-assets/track-1-recovery-example.gif)

![Example of center driving on Track 2](https://github.com/stephenvfg/behavioral-cloning/blob/master/writeup-assets/track-2-middle-example.gif)


![Model architecture](https://github.com/stephenvfg/behavioral-cloning/blob/master/writeup-assets/model.png)

![Example of view from the center cam](https://github.com/stephenvfg/behavioral-cloning/blob/master/writeup-assets/center-cam-normal.jpg)

![Example of the same center cam view, reversed](https://github.com/stephenvfg/behavioral-cloning/blob/master/writeup-assets/center-cam-reversed.jpg)




#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 25).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road ... 

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to ...

My first step was to use a convolution neural network model similar to the ... I thought this model might be appropriate because ...

In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set. I found that my first model had a low mean squared error on the training set but a high mean squared error on the validation set. This implied that the model was overfitting. 

To combat the overfitting, I modified the model so that ...

Then I ... 

The final step was to run the simulator to see how well the car was driving around track one. There were a few spots where the vehicle fell off the track... to improve the driving behavior in these cases, I ....

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

#### 3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.





