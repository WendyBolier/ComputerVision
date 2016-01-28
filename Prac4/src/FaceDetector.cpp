/*
* FaceDetector.cpp
*
*  Created on : Jan 12, 2016
*      Authors : Erik and Wendy
*/

#include <iostream>

#include <boost/filesystem.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "FaceDetector.h"

namespace fs = boost::filesystem;

namespace nl_uu_science_gmt
{
	FaceDetector::FaceDetector(const std::string &path_pos, const std::string &path_neg, const cv::Size &input_size,
		const int cell_size, const cv::Rect &crop_window) :
		m_pos_path(path_pos),
		m_neg_path(path_neg),
		m_model_size(input_size),
		m_cell_size(cell_size),
		m_crop(crop_window)
	{
	
	}

	FaceDetector::~FaceDetector()
	{

	}

	/**
	* Loads the image data from the disk given one or more paths
	*
	* INPUT   paths: a vector of strings to find training data
	* OUTPUT  images: a vector of CV_8U images read from the input paths
	* FLAG    is_positive (true): loading positive or negative images
	*         crop (true)       : crop the loaded images
	*         scale (false)     : scale the cropped images
	*/
	std::vector<int> FaceDetector::load(MatVec &trainingSamples, MatVec &validationSamples, const bool do_crop, const bool do_scale) {
		std::vector<std::string> positivesTraining, positivesValidation, negativesTraining, negativesValidation;

		//Read from the paths and load everything including the metadata paths into the PathVects
		cv::FileStorage fs(m_pos_path, cv::FileStorage::READ);
		fs["Training"] >> positivesTraining;
		fs["Validation"] >> positivesValidation;

		fs = cv::FileStorage(m_neg_path, cv::FileStorage::READ);
		fs["Training"] >> negativesTraining;
		fs["Validation"] >> negativesValidation;;

		if ((do_crop == true) && (do_scale == true))
		{
			std::cout << "Loading positive training images..." << std::endl;
			//Positive training images
			for each (std::string path in positivesTraining) {
				cv::Mat image = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
				cv::resize(image(m_crop), image, m_model_size);

				//Normalize the image and add the result
				trainingSamples.push_back(normalize(image));
				m_img_fns_pos.push_back(path);
				m_img_fns_pos_train.push_back(path);
			}
			std::cout << "Loading positive validation images..." << std::endl;
			//Positive validation images
			for each (std::string path in positivesValidation) {
				cv::Mat image = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
				cv::resize(image(m_crop), image, m_model_size);

				//Normalize the image and add the result
				validationSamples.push_back(normalize(image));
				m_img_fns_pos.push_back(path);
				m_img_fns_pos_validation.push_back(path);
			}

			std::vector<int> offsets;
			//Training offset
			offsets.push_back(trainingSamples.size());
			//Validation offset
			offsets.push_back(validationSamples.size());

			//initialize random seed:
			srand(time(NULL));
			int tooSmallCounter = 0;
			cv::Rect sampleWindow = m_crop;
			cv::Mat resized;
			std::cout << "Loading negative training images..." << std::endl;
			//Negative training images
			for each (std::string path in negativesTraining) {
				cv::Mat image = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);

				if (image.rows < m_crop.height || image.cols < m_crop.width) {
					tooSmallCounter++;
					continue;
				}
				for (int i = 0; i < 6; i++) {
					sampleWindow.x = rand() % (image.cols - sampleWindow.width + 1);
					sampleWindow.y = rand() % (image.rows - sampleWindow.height + 1);
					cv::resize(image(sampleWindow), resized, m_model_size);

					//Normalize the image and add the result
					trainingSamples.push_back(normalize(resized));
				}

				m_img_fns_neg.push_back(path);
				m_img_fns_neg_train.push_back(path);
				m_img_fns_neg_meta.push_back(path.replace(path.replace(23, 10, "Annotations").size() - 3, 3, "xml"));
			}
			std::cout << "Loading negative validation images..." << std::endl;
			//Negative validation images
			for each (std::string path in negativesValidation) {
				cv::Mat image = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);

				if (image.rows < m_crop.height || image.cols < m_crop.width) {
					tooSmallCounter++;
					continue;
				}
				for (int i = 0; i < 6; i++) {
					sampleWindow.x = rand() % (image.cols - sampleWindow.width + 1);
					sampleWindow.y = rand() % (image.rows - sampleWindow.height + 1);
					cv::resize(image(sampleWindow), resized, m_model_size);

					//Normalize the image and add the result
					validationSamples.push_back(normalize(resized));
				}

				m_img_fns_neg.push_back(path);
				m_img_fns_neg_validation.push_back(path);
				m_img_fns_neg_meta.push_back(path.replace(path.replace(23, 10, "Annotations").size() - 3, 3, "xml"));
			}

			std::cout << "Skipped " << tooSmallCounter << (tooSmallCounter == 1 ? " image because it is" : " images because they are") << " too small to crop..." << std::endl;
			return offsets;
		}
		return std::vector<int>();
	}

	cv::Mat FaceDetector::normalize(cv::Mat &image) {
		cv::Mat features = image.reshape(1, 1);
		features.convertTo(features, cv::DataType<float>::type, 1 / 255.0);
		cv::Mat mean, stddev;
		cv::meanStdDev(features, mean, stddev);

		features = features - mean;
		features = features / stddev;
		return features.reshape(1, image.rows);
	}

	void FaceDetector::svmFaces(const MatVec &trainingData, std::vector<int> offsets, SVMModel &model) {

	}

}