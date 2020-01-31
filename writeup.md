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
* [video.mp4](https://github.com/stephenvfg/behavioral-cloning/blob/master/video.mp4) which shows the vehicle successfully driving on track 1.
* [challenge-video.mp4](https://github.com/stephenvfg/behavioral-cloning/blob/master/challenge-video.mp4) which shows the vehicle attempting and having difficulty driving on track 2.

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

The data that I gathered for this model was split into two separated groups of training data and validation data (model.py lines 54-59). The model also contains a 25% dropout layer (model.py line 75) in order to reduce overfitting.

#### 3. Model parameter tuning

The model uses the mean squared error loss function with an Adam optimizer (model.py line 83). I did not manually tune the learning rate. 

I used 5 epochs for the model training since the improvement per epoch tapered off around the fourth or fifth epoch. Using a small batch size of 32 also helped with my model accuracy.

#### 4. Appropriate training data

The focus of my training data is to keep the vehicle in the center of the road. In situations where the vehicle veers close to the edge of the road I want it to understand how to react and recover. Since there are two different tracks for the vehicle to drive on, I eventually used data from both tracks.

After collecting training data I put together some additional functions to augment and preprocess the data prior to training. To ensure that the vehicle is properly equipped to react from its left side as well as its right side, I horizontally flipped all of my training images and multiplied the corresponding steering angles by -1 (model.py lines 39-41). This simulated more driving data as if the vehicle were performing the same correct turns in the opposite direction.

| Center camera view  | Same camera view, reversed |
| ------------------- | -------------------------- |
| (Steering angle) | (Steering angle) * -1.0 |
| ![Example of view from the center cam](https://github.com/stephenvfg/behavioral-cloning/blob/master/writeup-assets/center-cam-normal.jpg) | ![Example of the same center cam view, reversed](https://github.com/stephenvfg/behavioral-cloning/blob/master/writeup-assets/center-cam-reversed.jpg) |

For preprocessing, I normalized the training data by centering it around 0 between -0.5 and 0.5 (model.py line 65). I also cropped out unhelpful pixels from the bottom and top of the images (model.py line 66).

Finally I made sure to include data from all three vehicle cameras. To correct for the steering angles I either added or subtracted a 0.4 angle correction constant from the center steering angle for the left and right cameras respectively (model.py line 35).

| Left camera  | Center camera | Right camera |
| ------------ | ------------- | ------------ |
| (Steering angle) + 0.4 | (Steering angle) | (Steering angle) - 0.4 |
| ![Left camera angle](https://github.com/stephenvfg/behavioral-cloning/blob/master/writeup-assets/left.jpg) | ![Center camera angle](https://github.com/stephenvfg/behavioral-cloning/blob/master/writeup-assets/center.jpg) | ![Right camera angle](https://github.com/stephenvfg/behavioral-cloning/blob/master/writeup-assets/right.jpg) |

#### 5. Solution Design Approach

My overall strategy for deriving a model architecture was to find one that already works really well, and use that! In class we learned about several successful models when it comes to image recognition (AlexNet, GoogLeNet, etc). However the model that stuck out to me the most was the [NVIDIA deep learning model](https://devblogs.nvidia.com/deep-learning-self-driving-cars/) since it was designed specifically for self-driving cars.

From there I followed the standard approach of splitting my data into training and validation data sets, preprocessing and augmenting that data, training the model and then testing its accuracy on the validation data. 

To prevent overfitting, I modified the NVIDIA model by adding in one additional 25% dropout layer between the convolutional layers and the fully connected layers (model.py line 75). With this addition my training loss and validation became more consistent. My fifth training epoch produced a training loss value of 0.0717 and validation loss MSE value of 0.0692.

I initially trained my model using center lane driving data and recovery driving data from track one.

| Track one center lane driving | Track one recovery driving |
| ------------------- | -------------------------- |
| ![Example of center driving on Track 1](https://github.com/stephenvfg/behavioral-cloning/blob/master/writeup-assets/track-1-middle-example.gif) | ![Example of recovery driving on Track 1](https://github.com/stephenvfg/behavioral-cloning/blob/master/writeup-assets/track-1-recovery-example.gif) |

This enabled the autonomous vehicle to successfully drive around track one when the model was applied.

| Successful vehicle driving on track one |
| ------------------- |
| [Full footage from video output](https://github.com/stephenvfg/behavioral-cloning/blob/master/video.mp4) |
| ![Successful vehicle driving on track one](https://github.com/stephenvfg/behavioral-cloning/blob/master/writeup-assets/track1-success.gif) |

However this was not enough for the car to drive on track two. My vehicle immediately crashed on track two with just that data. To remediate I also took center lane driving footage and recovery footage from track two.

| Track two driving |
| ------------------- |
| ![Example of center driving on Track 2](https://github.com/stephenvfg/behavioral-cloning/blob/master/writeup-assets/track-2-middle-example.gif) |

As a result, my vehicle was able to get on the road and drive on track two! However it was not able to complete the entire track. It crashed during a sharp corner after recovering from a curve too sharply. In order to improve my model, my next approach would be to take additional training data from track two with an increased focus on double recovering from sharp curves and corners. I might also focus on collecting data from areas with the metal poles on track two - it's possible I didn't have enough similar data from those parts of the track.

| Unsuccessful vehicle driving on track two |
| ------------------- |
| [Full footage from video output](https://github.com/stephenvfg/behavioral-cloning/blob/master/challenge-video.mp4) |
| ![Unsuccessful vehicle driving on track two](https://github.com/stephenvfg/behavioral-cloning/blob/master/writeup-assets/track2-fail.gif) |

#### 6. Final Model Architecture

The final model architecture (model.py lines 61-80) consisted of a convolution neural network with the following layers and layer sizes:


Here is a visualization of the architecture:

![Model architecture](https://github.com/stephenvfg/behavioral-cloning/blob/master/writeup-assets/model.png)

#### 7. Creation of the Training Set & Training Process

To capture good driving behavior on both tracks, I recorded data for the following scenarios:

* Full laps around track one using center lane driving.
* Full lap around track one driving in the opposite direction.
* Partial lap around track one with a focus on recovery driving.
* Full lap around track two.
* Partial lap around track two with a focus on recovery driving.

Examples from my data collection process can be found in section 5 of this writeup.

I also augmented my data set by flipping the images to simulate driving in the opposite direction. Additionally I used data from all three cmaeras on the vehicle. Examples of this can be found in section 4 of this writeup.

Ultimately my dataset contained the following:

* Over 12,500 unique steering angle snapshots.
* 3 unique images from 3 different camera positions for each steering angle.
* 1 additional simulated image and angle for every collected image/angle combination.
* Total of over 75,000 image/steering angle pairs to train my model on.

The training data was preprocessed by normalizing the data around 0 between -0.5 and 0.5 and then cropping the data to remove useless information at the top and bottom. Then the data was shuffled and split so that 20% of the data would be used for validation.

I used an Adam optimizer so I did not need to worry about setting a learning rate for the training process. The training loss improvement tapered off after the fourth or fifth epoch so I settled on 5 epochs for my training process.

All of this let to a sucessful autonomous lap around track one and an "almost there" lap around track two!
