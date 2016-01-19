#include <cstdlib>
#include <string>
#include <iostream>
#include <sstream>
#include <sys/stat.h>

#include <boost/lexical_cast.hpp>
#include <boost/filesystem.hpp>
#include <boost/foreach.hpp>
#include <boost/property_tree/xml_parser.hpp>
#include <boost/property_tree/ptree.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "FaceDetector.h"

using namespace nl_uu_science_gmt;
using namespace cv;
namespace fs = boost::filesystem;

std::string pathTrainingPositive = "data\\Training Positive";
std::string pathTrainingNegative = "data\\Training Negative";
std::string pathTrainingNegativeMetadata = "data\\Training Negative\\Annotations";
std::string pathTrainingNegativeImages = "data\\Training Negative\\JPGImages";
std::string pathUsableNegative = "Filtered Negatives.xml";

inline bool fileExists(const std::string& name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}

void sortNegatives() {
	FileStorage fs(pathUsableNegative, FileStorage::WRITE);

	// First check if the start path exists
	if (!fs::exists(pathTrainingNegativeMetadata) || !fs::is_directory(pathTrainingNegativeMetadata))
	{
		std::cout << "Given path not a directory or does not exist" << std::endl;
		return;
	}

	// Create iterators for iterating all entries in the directory
	fs::directory_iterator it(pathTrainingNegativeMetadata);    // Directory iterator at the start of the directory
	fs::directory_iterator end;									// Directory iterator by default at the end

	std::cout << "Detecting all negative images without people...";

	// Iterate all entries in the directory
	while (it != end)
	{
		boost::filesystem::path currentPath = it->path();
		std::cout << "Checking " << currentPath.string() << std::endl;
		if (boost::filesystem::is_regular_file(currentPath)) {
			//foreach image in JPEGImages
			std::string keyname = "name";
			std::string value = "person";
			std::ifstream ifs(currentPath.string(), std::ifstream::in);
			boost::property_tree::ptree pt;
			boost::property_tree::read_xml(ifs, pt);
			bool has_person = false;
			BOOST_FOREACH(boost::property_tree::ptree::value_type const& node,
				pt.get_child("annotation.object")) {
				if (node.first == keyname &&
					node.second.get_value<std::string>() == value) {
					has_person = true;
					break;
				}
			}
			if (!has_person) {
				fs << "File" << currentPath.string();
			}
		}

		// Next directory entry
		it++;
	}
}

int main(int argc, char** argv) {
	cv::Rect rect(76, 94, 98, 98);
	FaceDetector detector("data\\Training Positive", "data\\Training Negative", cv::Size(20, 20), 1, rect, 1000, 2);
	PathVec pv;
	MatVec images;
	detector.load(pv, images, true, true, true);

	if (!fileExists(pathUsableNegative)) {
		sortNegatives();
	}

	/*
	TODO:
	- If it doesn't exist yet, set up matrices with a random 80%/20% test+validation data separation and write it to a file
		cv::FileStorage fs(s.outputFileName, FileStorage::WRITE); fs << "DataMatrix" << matrix;
	- 
	*/

	return EXIT_SUCCESS;
}