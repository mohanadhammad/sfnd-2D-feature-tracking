# SFND 2D Feature Tracking

<img src="images/keypoints.png" width="820" height="248" />

The idea of the camera course is to build a collision detection system - that's the overall goal for the Final Project. As a preparation for this, you will now build the feature tracking part and test various detector / descriptor combinations to see which ones perform best. This mid-term project consists of four parts:

* First, you will focus on loading images, setting up data structures and putting everything into a ring buffer to optimize memory load. 
* Then, you will integrate several keypoint detectors such as HARRIS, FAST, BRISK and SIFT and compare them with regard to number of keypoints and speed. 
* In the next part, you will then focus on descriptor extraction and matching using brute force and also the FLANN approach we discussed in the previous lesson. 
* In the last part, once the code framework is complete, you will test the various algorithms in different combinations and compare them with regard to some performance measures. 

See the classroom instruction and code comments for more details on each of these parts. Once you are finished with this project, the keypoint matching part will be set up and you can proceed to the next lesson, where the focus is on integrating Lidar points and on object detection using deep-learning. 

## Dependencies for Running Locally
* cmake >= 2.8
  * All OSes: [click here for installation instructions](https://cmake.org/install/)
* make >= 4.1 (Linux, Mac), 3.81 (Windows)
  * Linux: make is installed by default on most Linux distros
  * Mac: [install Xcode command line tools to get make](https://developer.apple.com/xcode/features/)
  * Windows: [Click here for installation instructions](http://gnuwin32.sourceforge.net/packages/make.htm)
* OpenCV >= 4.1
  * This must be compiled from source using the `-D OPENCV_ENABLE_NONFREE=ON` cmake flag for testing the SIFT and SURF detectors.
  * The OpenCV 4.1.0 source code can be found [here](https://github.com/opencv/opencv/tree/4.1.0)
* gcc/g++ >= 5.4
  * Linux: gcc / g++ is installed by default on most Linux distros
  * Mac: same deal as make - [install Xcode command line tools](https://developer.apple.com/xcode/features/)
  * Windows: recommend using [MinGW](http://www.mingw.org/)

## Basic Build Instructions

1. Clone this repo.
2. Make a build directory in the top level directory: `mkdir build && cd build`
3. Compile: `cmake .. && make`
4. Run it: `./2D_feature_tracking`.

## Rubrics:
1. MP.1 Data Buffer Optimization

This is achieved by checking the size of the buffer directly after inserting new data and if exceeds the defined maximum size then remove the first element in the buffer using the method erase() in vector library.

2. MP.2 Keypoint Detection

Implemented from OpenCV library the detectors HARRIS, FAST, BRISK, ORB, AKAZE, and SIFT and make them selectable by setting a string accordingly. I have added also code to measure the execution time of the detection algorithm and display on consol the number of keypoints detected.

3. MP.3 Keypoint Removal

Remove all keypoints outside of a pre-defined rectangle and only use the keypoints within the rectangle for further processing. I achieved this by using "cv::Rect" and using the method "contains()" which is useful to check points inside the defined rectangle.

4. MP.4 Keypoint Descriptors

Implemented from OpenCV the descriptors BRIEF, ORB, FREAK, AKAZE and SIFT and make them selectable by setting a string accordingly. I have added also code to measure the execution time.

5. MP.5 Descriptor Matching

Implemented FLANN matching as well as k-nearest neighbor selection. Both methods are selectable using the respective strings in the main function. I have a hack to overwrite the descriptor type of the current and previous images to "CV_32F".
In the Brute Force (BF) Matching code I have added a code to select the norm type between "cv::NORM_HAMMING" and "cv::NORM_L2" based on the feature type if its binary [ex: ORB, FREAK] or not [ex: SIFT].

6. MP.6 Descriptor Distance Ratio

Used the K-Nearest-Neighbor matching to implement the descriptor distance ratio test, which looks at the ratio of best vs. second-best match to decide whether to keep an associated pair of keypoints. The ratio I have used is 0.8.

7. MP.7 Performance Evaluation 1

Count the number of keypoints on the preceding vehicle for all 10 images and take note of the distribution of their neighborhood size. Do this for all the detectors you have implemented.

**[Check the generated csv files for all combination in the doc folder]**

8. MP.8 Performance Evaluation 2

Count the number of matched keypoints for all 10 images using all possible combinations of detectors and descriptors. In the matching step, the BF approach is used with the descriptor distance ratio set to 0.8.

**[Check the generated csv files for all combination in the doc folder]**

9.  MP.9 Performance Evaluation 3

Log the time it takes for keypoint detection and descriptor extraction. The results must be entered into a spreadsheet and based on this data, the TOP3 detector / descriptor combinations must be recommended as the best choice for our purpose of detecting keypoints on vehicles.

**[Check the generated csv files for all combination in the doc folder]**

The AKAZE descriptor only works in combination with AKAZE detector and the same also for SIFT.

##### Top3 detector / descriptor combinations:
Although ORB + FREAK combination has most of the good matched keypoints but it took long execution time with average of 53 ms but on the other hand it match with high percentage of accuracy 330 keypoints out of 7683 detected keypoints (considering also the filtration for the preceding vehicle) but on the other hand the below recommendations have the fastest execution time compromized with the good matched keypoints which are recommended for real time systems.

1. FAST + BRIEF

Total execution time (ms) = 2.56 ms
Keypoints detected (Average) = 1787
Good Matched Keypoints (Average) = 122 [including filtration for preceding vehicle]

2. FAST + ORB

Total execution time (ms) = 5.62 ms
Keypoints detected (Average) = 1787
Good Matched Keypoints (Average) = 120 [including filtration for preceding vehicle]

3. FAST + BRISK

Total execution time (ms) = 3.43 ms
Keypoints detected (Average) = 1787
Good Matched Keypoints (Average) = 100 [including filtration for preceding vehicle]
