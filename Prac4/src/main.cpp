#include <cstdlib>
#include <string>
#include "FaceDetector.h"
#include <iostream>

using namespace nl_uu_science_gmt;

int main(
	int argc, char** argv)
{
	cv::Rect rect(76, 94, 98, 98);
	FaceDetector detector("data\Training Positive", "data\Training Negative", cv::Size(20, 20), 1, rect, 1000, 2);
	PathVec pv;
	MatVec images;
	detector.load(pv, images, true, true, true);

	return EXIT_SUCCESS;
}