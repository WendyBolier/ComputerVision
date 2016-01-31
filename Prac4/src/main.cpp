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
#include "MySVM.h"


/* Toggles */
//#define _TestVariables


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

#ifndef _TestVariables
	cv::namedWindow("Display");
#endif
	cv::Rect modelWindow(76, 90, 100, 100);

	int modelSize = 20;
	double threshold = 0;
	float scaleFactor = 1.3;

#ifdef _TestVariables
	cv::FileStorage variableTestResults("VariableResults.xml", cv::FileStorage::WRITE);
	for (modelSize = 10; modelSize <= 20; modelSize += 5) {
		for (threshold = -1; threshold <= 2; threshold += 0.25) {
			for (scaleFactor = 1.1; scaleFactor <= 1.5; scaleFactor += 0.25) {
				variableTestResults << "modelSize" << modelSize;
				variableTestResults << "threshold" << std::to_string(threshold);
				variableTestResults << "scaleFactor" << std::to_string(scaleFactor);
				std::cout << "Testing modelSize = " << modelSize << ", threshold = " << threshold << ", scaleFactor = " << scaleFactor << std::endl;
#endif

				cv::Size resize(modelSize, modelSize);
				FaceDetector detector(pathPositivesToUse, pathUsableNegative, resize, 1, modelWindow);

				cv::Mat trainingSamples, validationSamples;
				//Load the images, normalized, and get the negative offsets in the training and validation sets, respectively
				std::vector<int> offsets = detector.load(trainingSamples, validationSamples, true, true);

				//Train the model
				MySVM svm;
				SVMModel svmModel;
				detector.svmFaces(trainingSamples, validationSamples, svm, offsets, svmModel);

				int finalImage = 7;

#ifdef _TestVariables
				finalImage = 6;
#endif
				for (int imgidx = 1; imgidx <= finalImage; imgidx++) {
					//There is no img3.jpg
					if (imgidx == 3) imgidx++;
					std::string pathSegment = imgidx == 7 ? "-test.png" : std::to_string(imgidx) + ".jpg";

					//Create the pyramid from the source image
					ImagePyramid pyramid;
					cv::Mat img = cv::imread(std::string("data\\Test\\img") + pathSegment, CV_LOAD_IMAGE_GRAYSCALE);
					detector.createPyramid(scaleFactor, img, pyramid);

					//Decide the PDF per layer
					Size size = img.size();
					CandidateVec candidates;
					for (int l = 0; l < pyramid.size(); l++) {
						detector.convolve(svmModel, svm, pyramid[l]);
						detector.positionalContent(pyramid[l], threshold, candidates);
					}
					detector.nonMaximaSuppression(size, candidates);

					// draw final candidates on the image
					img = imread(std::string("data\\Test\\img") + pathSegment);
					for (int i = 0; i < candidates.size(); i++) {
						rectangle(img, candidates[i].img_roi, cv::Scalar(0, 255, 255), 1, 8, 0);
					}
					imwrite(std::string("TestResults") + pathSegment, img);

					if (img.cols > 900 || img.rows > 900) {
						cv::resize(img, img, cv::Size(img.cols / 2, img.rows / 2));
					}

					//The last one doesn't have ground truth data
					if (imgidx != 7) {
						//Find true/false positives
						FileStorage fs(std::string("data\\Test\\img") + pathSegment.replace(pathSegment.size() - 3, 3, "xml"), FileStorage::READ);
						vector<int> groundTruth;
						fs["ground_truth"] >> groundTruth;
						int truePositives = 0, falseNegatives = 0, guesses = candidates.size();

						for (int i = 0; i < groundTruth.size(); i += 4) {
							bool faceFound = false;
							cv::Rect faceBox(groundTruth[i], groundTruth[i + 1], groundTruth[i + 2], groundTruth[i + 3]);
							for (int c = 0; c < candidates.size(); c++) {
								cv::Rect box = candidates[c].img_roi & faceBox;
								//Check whether the union/overlap between our match and the ground truth match is a big enoug part of both
								//We'll call that a TP
								if (box.area() > 0.5 * candidates[c].img_roi.area() & box.area() > 0.5 * faceBox.area()) {
									truePositives++;
									faceFound = true;
									//We stop checking against a candidate if it has been a TP for a match already
									candidates.erase(candidates.begin() + c);
									break;
								}
							}
							if (!faceFound) {
								falseNegatives++;
							}
						}
						std::cout << "Out of " << (falseNegatives + truePositives) << " positives, we found " << truePositives << " with a total of " << guesses << " guesses." << std::endl;

#ifdef _TestVariables
						// Write precision, recall and the F-score for the current configuration
						float precision = (float)truePositives / guesses;
						float recall = (float)truePositives / (truePositives + falseNegatives);
						float fScore = 2 * precision * recall / (precision + recall);
						variableTestResults << "Precision" + std::to_string(imgidx) << std::to_string(precision);
						variableTestResults << "Recall" + std::to_string(imgidx) << std::to_string(recall);
						variableTestResults << "F-score" + std::to_string(imgidx) << std::to_string(fScore);
#endif
					}

#ifndef _TestVariables
					std::cout << "Showing img" << pathSegment.replace(pathSegment.size() - 3, 3, "jpg") << ". Press any key to continue..." << std::endl;
					imshow("Display", img);
					waitKey();
#endif
				}

#ifdef _TestVariables
			}
		}
	}
#endif

	return EXIT_SUCCESS;
}