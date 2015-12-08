/***
* Header file for added functionality during Assignment 1
*
* Written by Erik Molenaar and Wendy Bolier
*
* Other functions not defined here come directly from the calibration tutorial on the OpenCV page.
* Further work we did can be found in main(), like checking for the existence of the calibration file and loading it,
* changing some of the displayed text, making recalibration possible (setting settings correctly), managing the cube's location
* and everything within the "if (mode == CALIBRATED && s.showUndistorted)"-block.
****/

#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>
#include <sys/stat.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace cv;
using namespace std;

void calibrateCamera(int video);
void saveCameraParams(Mat& cameraMatrix, Mat& distCoeffs, int video);

//Loads the camera parameters using the output file from the calibration
//Commented because the definition is not allowed here due to Settings being defined within the .cpp file
//static void loadCameraParams(Settings& s, Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs, const vector<float>& reprojErrs, double totalAvgErr);