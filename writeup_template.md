#**Behavioral Cloning** 

##Writeup Template

###You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report


[//]: # (Image References)

[centering]: ./examples/center.png "centering.png"
[recover1]: ./examples/recover1.png "recover1.png"
[recover2]: ./examples/recover2.png "recover2.png"
[recover3]: ./examples/recover3.png "recover3.png"
[fail_lenet]: ./examples/placeholder_small.png "fail-lenet.png"
[nvidia-track2-crash1]: ./examples/nvidia-track2-crash1.png "nvidia-track2-crash1.png"
[nvidia-track2-crash2]: ./examples/nvidia-track2-crash2.png "nvidia-track2-crash2.png"

## Rubric Points
###Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
###Files Submitted & Code Quality

####1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md or writeup_report.pdf summarizing the results

####2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model-nvidia3.h5
```

####3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

###Model Architecture and Training Strategy

####1. An appropriate model architecture has been employed

There are a number of different models as introduced in the class as well as the model from commaAI. The models can be chosen using the --models command line argument.

####2. Attempts to reduce overfitting in the model

The commaAI model contains dropout layers, the others don't.

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 10-16). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

####3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually.

####4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road. The original udacity dataset is augmented with driving in the center one lap, driving in the center one backwards lap, recovering in the regular direction for one lap, recovering in the regular direction for one backwards lap.

For details about how I created the training data, see the next section. 

###Model Architecture and Training Strategy

####1. Solution Design Approach

The overall strategy for deriving a model architecture was to watch the classroom videos.

My first step was to use a convolution neural network model similar to LeNet. I anticipated this model might not be appropriate. The model did, however, drive a fair bit over the bridge. Ultimately it drove into the wild. I think the training data contains a small track where I drive into the wild in order to take a shortcut. However, the model seems to not follow the shortcut precisely enough so it fails to recover from the situation.

The commai model (without cropping) wasn't even able to drive over the first bridge.

The nvidia model performs quite good and manages to do the full first track. I tried it with the second track and it got around a little bit.

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

####2. Final Model Architecture

The final model architecture is a convolutional neral net defined in modelNvidia. It contains a lambda to normalize and mean center the image data. The image is cropped to focus on the relevant bit, which is the street. Three 5x5 convolutions and two 3x3 convolutions with relu activations are used. The number of hidden nodes is much larger than in LeNet with 24,36,48,64,64 nodes. Finally multiple fully connected layers (FC-100, FC-50, FC-10, FC-1) reduce the output to a single one.


####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][centering]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to recover. These images show what a recovery looks like at random places:

![alt text][recover1]
![alt text][recover2]
![alt text][recover3]

I repeated this process a few times on the first track driving a single lap both in the normal direction as well as from the other direction.

To augment the data sat, I flipped the images and angles thinking that this would double the amount of usable training data while also balancing left turns vs. right turns. Furthermore all three cameras are used with a compensation value of 0.2. 

After the collection process, I had 16812*3*2 data samples.

The lenet model would drive over the bridge and turn into the bush. I think when I recorded the data I did on purpose drive there to make a shortcut. I am not sure anymore. This would explain the behaviour. Unfortunately the car doesn't find its way out again. Presumably because it barely can see the road anymore and ends up in a steep angle that it hasn't seen before.

The commai model surprisingly fails to take the curve before the bridge already.

The nvidia model drives the whole lap! Well done.

When I tried the nvidia model on the second track it manages to drive up the first hill a little bit before it crashes.

![alt text][nvidia-track2-crash1]

I then recorded center driving the second lap fully for once.
This would give me 18302*3*2 data samples.
The model would now drive much further but get stuck driving up a hill.

![alt text][nvidia-track2-crash2]

Specifically training this scenario gave me a new model which then would drive even worse at the very beginning. I gave up on track 2.
