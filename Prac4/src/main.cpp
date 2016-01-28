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
std::string pathTrainingNegativeImages = "data\\Training Negative\\JPEGImages";
std::string pathUsableNegative = "Filtered Negatives.xml";
std::string pathPositivesToUse = "Filtered Positives.xml";

inline bool fileExists(const std::string& name) {
	struct stat buffer;
	return (stat(name.c_str(), &buffer) == 0);
}

void sortNegatives() {
	FileStorage fs(pathUsableNegative, FileStorage::WRITE);

	int counterTraining = 0, counterValidation = 0;

	// First check if the start path exists
	if (!fs::exists(pathTrainingNegativeMetadata) || !fs::is_directory(pathTrainingNegativeMetadata)) {
		std::cout << "Given path (" << pathTrainingNegativeMetadata << ") not a directory or does not exist" << std::endl;
		return;
	}

	// Create iterators for iterating all entries in the directory
	fs::directory_iterator it(pathTrainingNegativeMetadata);    // Directory iterator at the start of the directory
	fs::directory_iterator end;									// Directory iterator by default at the end

	std::cout << "Detecting all negative images without people...";

	//initialize random seed:
	srand(time(NULL));

	vector<string> trainingFiles, validationFiles;

	std::string keyobj = "object";
	std::string keyname = "name";
	std::string value = "person";

	// Iterate all entries in the directory
	while (it != end) {
		boost::filesystem::path currentPath = it->path();
		std::cout << "Checking " << currentPath.string() << std::endl;
		if (boost::filesystem::is_regular_file(currentPath)) {
			//foreach image in JPEGImages
			std::ifstream ifs(currentPath.string(), std::ifstream::in);
			boost::property_tree::ptree pt;
			boost::property_tree::read_xml(ifs, pt);
			bool has_person = false;
			BOOST_FOREACH(boost::property_tree::ptree::value_type const& node,
				pt.get_child("annotation")) {
				if (node.first == keyobj
					&& node.second.get_child(keyname).get_value<std::string>() == value) {
					has_person = true;
					break;
				}
			}
			if (!has_person) {
				string a = currentPath.string();

				if (rand() % 100 + 1 <= 20) {
					//A 20% chance to be validation data
					validationFiles.push_back(a.replace(0, 34, pathTrainingNegativeImages).replace(a.size() - 4, 3, "jpg"));
					counterValidation++;
				}
				else {
					//An 80% chance to be training data
					trainingFiles.push_back(a.replace(0, 34, pathTrainingNegativeImages).replace(a.size() - 4, 3, "jpg"));
					counterTraining++;
				}
			}
		}

		// Next directory entry
		it++;
	}

	fs << "Training" << trainingFiles;
	fs << "Validation" << validationFiles;
	std::cout << "Selected " << counterTraining << " negative training images and " << counterValidation << " negative validation images." << std::endl;
}

void sortPositives() {
	FileStorage fs(pathPositivesToUse, FileStorage::WRITE);

	// First check if the start path exists
	if (!fs::exists(pathTrainingPositive) || !fs::is_directory(pathTrainingPositive)) {
		std::cout << "Given path (" << pathTrainingPositive << ") not a directory or does not exist" << std::endl;
		return;
	}

	int counterTraining = 0, counterValidation = 0;

	// Create iterators for iterating all entries in the directory
	fs::directory_iterator iterator(pathTrainingPositive);    // Directory iterator at the start of the directory
	fs::directory_iterator end;							// Directory iterator by default at the end

	std::cout << "Selecting positive images and sorting them between training and validation...";

	//initialize random seed:
	srand(time(NULL));

	vector<string> trainingFiles, validationFiles;

	// Iterate all entries in the directory
	while (iterator != end) {
		boost::filesystem::path currentPath = iterator->path();
		std::cout << "Checking " << currentPath.string() << std::endl;

		// Create iterators for iterating all entries in the subdirectory
		fs::directory_iterator subIterator(currentPath);    // Subdirectory iterator at the start of the directory
		fs::directory_iterator subEnd;						// Subdirectory iterator by default at the end

		//Loop over all the images
		while (subIterator != subEnd) {
			boost::filesystem::path imagePath = subIterator->path();

			//Half of the positive images, with 6 samples per negative, makes 3 times as many negatives as positives
			if (rand() % 100 + 1 <= 50) {
				if (rand() % 100 + 1 <= 20) {
					//A 20% chance to be validation data
					validationFiles.push_back(imagePath.string());
					counterValidation++;
				}
				else {
					//An 80% chance to be training data
					trainingFiles.push_back(imagePath.string());
					counterTraining++;
				}
			}
			subIterator++;
		}

		// Next directory entry
		iterator++;
	}

	fs << "Training" << trainingFiles;
	fs << "Validation" << validationFiles;
	std::cout << "Selected " << counterTraining << " positive training images and " << counterValidation << " positive validation images." << std::endl;
}

int main(int argc, char** argv) {
	if (!fileExists(pathUsableNegative)) {
		sortNegatives();
	}
	if (!fileExists(pathPositivesToUse)) {
		sortPositives();
	}
	if (!fileExists(pathUsableNegative) || !fileExists(pathPositivesToUse)) {
		return EXIT_FAILURE;
	}

	cv::namedWindow("Display");
	cv::Rect modelWindow(76, 90, 100, 100);
	cv::Size resize(20, 20);
	FaceDetector detector(pathPositivesToUse, pathUsableNegative, resize, 1, modelWindow);

	MatVec trainingSamples, validationSamples;
	//Load the images, normalized, and get the negative offsets in the training and validation sets, respectively
	std::vector<int> offsets = detector.load(trainingSamples, validationSamples, true, true);

	SVMModel svmModel;
	detector.svmFaces(trainingSamples, offsets, svmModel);
	return EXIT_SUCCESS;
}