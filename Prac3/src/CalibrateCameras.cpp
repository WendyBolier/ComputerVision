#ifndef _CRT_SECURE_NO_WARNINGS
# define _CRT_SECURE_NO_WARNINGS
#endif

#include <iostream>
#include <sstream>
#include <time.h>
#include <stdio.h>
#include <sys/stat.h>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "utilities\General.h"
#include "CalibrateCameras.h"


using namespace nl_uu_science_gmt;
using namespace cv;
using namespace std;


class Settings
{
public:
	Settings() : goodInput(false) {}
	enum Pattern { NOT_EXISTING, CHESSBOARD, CIRCLES_GRID, ASYMMETRIC_CIRCLES_GRID };
	enum InputType { INVALID, CAMERA, VIDEO_FILE, IMAGE_LIST };


	void read(const FileNode& node)                          //Read serialization for this class
	{
		node["BoardSize_Width"] >> boardSize.width;
		node["BoardSize_Height"] >> boardSize.height;
		node["Calibrate_Pattern"] >> patternToUse;
		node["Square_Size"] >> squareSize;
		node["Calibrate_NrOfFrameToUse"] >> nrFrames;
		node["Calibrate_FixAspectRatio"] >> aspectRatio;
		node["Calibrate_AssumeZeroTangentialDistortion"] >> calibZeroTangentDist;
		node["Calibrate_FixPrincipalPointAtTheCenter"] >> calibFixPrincipalPoint;
		node["Input_FlipAroundHorizontalAxis"] >> flipVertical;
		node["Input_Delay"] >> delay;
		interprate();
	}
	void interprate()
	{
		goodInput = true;
		if (boardSize.width <= 0 || boardSize.height <= 0)
		{
			cerr << "Invalid Board size: " << boardSize.width << " " << boardSize.height << endl;
			goodInput = false;
		}
		if (squareSize <= 10e-6)
		{
			cerr << "Invalid square size " << squareSize << endl;
			goodInput = false;
		}
		if (nrFrames <= 0)
		{
			cerr << "Invalid number of frames " << nrFrames << endl;
			goodInput = false;
		}

		flag = 0;
		if (calibFixPrincipalPoint) flag |= CV_CALIB_FIX_PRINCIPAL_POINT;
		if (calibZeroTangentDist)   flag |= CV_CALIB_ZERO_TANGENT_DIST;
		if (aspectRatio)            flag |= CV_CALIB_FIX_ASPECT_RATIO;


		calibrationPattern = NOT_EXISTING;
		if (!patternToUse.compare("CHESSBOARD")) calibrationPattern = CHESSBOARD;
		if (calibrationPattern == NOT_EXISTING)
		{
			cerr << " Inexistent camera calibration mode: " << patternToUse << endl;
			goodInput = false;
		}
		atImageList = 0;

	}
	Mat nextImage()
	{
		Mat result;
		if (inputCapture.isOpened())
		{
			Mat view0;
			inputCapture >> view0;
			view0.copyTo(result);
		}
		else if (atImageList < (int)imageList.size())
			result = imread(imageList[atImageList++], CV_LOAD_IMAGE_COLOR);

		return result;
	}

	static bool readStringList(const string& filename, vector<string>& l)
	{
		l.clear();
		FileStorage fs(filename, FileStorage::READ);
		if (!fs.isOpened())
			return false;
		FileNode n = fs.getFirstTopLevelNode();
		if (n.type() != FileNode::SEQ)
			return false;
		FileNodeIterator it = n.begin(), it_end = n.end();
		for (; it != it_end; ++it)
			l.push_back((string)*it);
		return true;
	}
public:
	Size boardSize;            // The size of the board -> Number of items by width and height
	Pattern calibrationPattern;// One of the Chessboard, circles, or asymmetric circle pattern
	float squareSize;          // The size of a square in your defined unit (point, millimeter,etc).
	int nrFrames;              // The number of frames to use from the input for calibration
	float aspectRatio;         // The aspect ratio
	int delay;                 // In case of a video input
	bool bwritePoints;         //  Write detected feature points
	bool bwriteExtrinsics;     // Write extrinsic parameters
	bool calibZeroTangentDist; // Assume zero tangential distortion
	bool calibFixPrincipalPoint;// Fix the principal point at the center
	bool flipVertical;          // Flip the captured images around the horizontal axis
	string outputFileName;      // The name of the file where to write
	bool showUndistorted;       // Show undistorted images after calibration
	string input;               // The input ->



	int cameraID;
	vector<string> imageList;
	int atImageList;
	VideoCapture inputCapture;
	InputType inputType;
	bool goodInput;
	int flag;

private:
	string patternToUse;


};

static void read(const FileNode& node, Settings& x, const Settings& default_value = Settings())
{
	if (!node.empty())
		x.read(node);
	else
		x = default_value;
}

inline bool fileExists(const std::string& name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}

static double computeReprojectionErrors(const vector<vector<Point3f> >& objectPoints,
	const vector<vector<Point2f> >& imagePoints,
	const vector<Mat>& rvecs, const vector<Mat>& tvecs,
	const Mat& cameraMatrix, const Mat& distCoeffs,
	vector<float>& perViewErrors)
{
	vector<Point2f> imagePoints2;
	int i, totalPoints = 0;
	double totalErr = 0, err;
	perViewErrors.resize(objectPoints.size());

	for (i = 0; i < (int)objectPoints.size(); ++i)
	{
		projectPoints(Mat(objectPoints[i]), rvecs[i], tvecs[i], cameraMatrix,
			distCoeffs, imagePoints2);
		err = norm(Mat(imagePoints[i]), Mat(imagePoints2), CV_L2);

		int n = (int)objectPoints[i].size();
		perViewErrors[i] = (float)std::sqrt(err*err / n);
		totalErr += err*err;
		totalPoints += n;
	}

	return std::sqrt(totalErr / totalPoints);
}

static void calcBoardCornerPositions(Size boardSize, float squareSize, vector<Point3f>& corners,
	Settings::Pattern patternType /*= Settings::CHESSBOARD*/)
{
	corners.clear();

	for (int i = 0; i < boardSize.height; ++i)
	for (int j = 0; j < boardSize.width; ++j)
		corners.push_back(Point3f(float(j*squareSize), float(i*squareSize), 0));
}

static bool runCalibration(Settings& s, Size& imageSize, Mat& cameraMatrix, Mat& distCoeffs,
	vector<vector<Point2f> > imagePoints, vector<Mat>& rvecs, vector<Mat>& tvecs,
	vector<float>& reprojErrs, double& totalAvgErr)
{

	cameraMatrix = Mat::eye(3, 3, CV_64F);
	if (s.flag & CV_CALIB_FIX_ASPECT_RATIO)
		cameraMatrix.at<double>(0, 0) = 1.0;

	distCoeffs = Mat::zeros(8, 1, CV_64F);

	vector<vector<Point3f> > objectPoints(1);
	calcBoardCornerPositions(s.boardSize, s.squareSize, objectPoints[0], s.calibrationPattern);

	objectPoints.resize(imagePoints.size(), objectPoints[0]);

	//Find intrinsic and extrinsic camera parameters
	double rms = calibrateCamera(objectPoints, imagePoints, imageSize, cameraMatrix,
		distCoeffs, rvecs, tvecs, s.flag | CV_CALIB_FIX_K4 | CV_CALIB_FIX_K5);

	cout << "Re-projection error reported by calibrateCamera: " << rms << endl;

	bool ok = checkRange(cameraMatrix) && checkRange(distCoeffs);

	totalAvgErr = computeReprojectionErrors(objectPoints, imagePoints,
		rvecs, tvecs, cameraMatrix, distCoeffs, reprojErrs);

	return ok;
}

// Print camera parameters to the output file
void saveCameraParams(Mat& cameraMatrix, Mat& distCoeffs, int video)
{
	FileStorage fs("data\\cam" + to_string(video + 1) + "\\" + General::IntrinsicsFile, FileStorage::WRITE);

	fs << "CameraMatrix" << cameraMatrix;
	fs << "DistortionCoeffs" << distCoeffs;

	fs.release();
}

void calibrateCamera(int video)
{
	Settings s;
	const string inputSettingsFile = "default.xml";
	FileStorage fs(inputSettingsFile, FileStorage::READ); // Read the settings
	if (!fs.isOpened())
	{
		cout << "Could not open the configuration file: \"" << inputSettingsFile << "\"" << endl;
	}
	fs["Settings"] >> s;
	fs.release();                                         // close Settings file

	if (!s.goodInput)
	{
		cout << "Invalid input detected. Application stopping. " << endl;
	}

	cout << "Calibrating camera " + to_string(video + 1) + "..." << endl;

	vector<vector<Point2f> > imagePoints;
	Mat cameraMatrix, distCoeffs;
	Size imageSize;
	clock_t prevTimestamp = 0;
	const Scalar RED(0, 0, 255), GREEN(0, 255, 0);
	const char ESC_KEY = 27;

	s.inputCapture.open("data\\cam" + to_string(video + 1) + "\\" + General::IntrinsicsVideo);

	for (int i = 0;; ++i)
	{
		Mat view;
		bool blinkOutput = false;

		view = s.nextImage();

		if (view.empty() || imagePoints.size() >= s.nrFrames)          // If no more images then run calibration, save and stop loop.
		{
			if (imagePoints.size() > 0)
			{
				vector<Mat> rvecs, tvecs;
				vector<float> reprojErrs;
				double totalAvgErr = 0;

				bool ok = runCalibration(s, imageSize, cameraMatrix, distCoeffs, imagePoints, rvecs, tvecs,
					reprojErrs, totalAvgErr);
				saveCameraParams(cameraMatrix, distCoeffs, video);
				break;
			}
		}


		imageSize = view.size();  // Format input image.
		if (s.flipVertical)    flip(view, view, 0);

		vector<Point2f> pointBuf;
		bool found = findChessboardCorners(view, s.boardSize, pointBuf,
			CV_CALIB_CB_ADAPTIVE_THRESH | CV_CALIB_CB_FAST_CHECK | CV_CALIB_CB_NORMALIZE_IMAGE);
		if (found)                // If done with success,
		{
			// improve the found corners' coordinate accuracy for chessboard
			Mat viewGray;
			cvtColor(view, viewGray, COLOR_BGR2GRAY);
			cornerSubPix(viewGray, pointBuf, Size(11, 11),
				Size(-1, -1), TermCriteria(CV_TERMCRIT_EPS + CV_TERMCRIT_ITER, 30, 0.1));

			if ((!s.inputCapture.isOpened() || clock() - prevTimestamp > s.delay*1e-3*CLOCKS_PER_SEC))
			{
				imagePoints.push_back(pointBuf);
				cout << to_string(imagePoints.size()) + "/" + to_string(s.nrFrames) + " frames found..." << endl;
				//prevTimestamp = clock();
				blinkOutput = s.inputCapture.isOpened();
				
				//Skip 75 frames or 1.5 seconds
				for (int j = 0; j <= 75; j++) {
					view = s.nextImage();
				}
			}
		}
	}
}