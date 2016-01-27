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
	void FaceDetector::prepare(const MatVec &pos_examples, const MatVec &neg_examples, const double factor,
		cv::Mat &Xt32F, cv::Mat &Xv32F, cv::Mat &Lt16S, cv::Mat &Lv16S)
	{
		int decider;
		// loop through all positive images
		for (int i = 0; i < pos_examples.size(); i++)
		{
			// random number between 0 and 9
			decider = rand() % 10;
			// if decider is 8 or 9, the image is added to the Validation Data
			if (decider > 7)
			{
				//Xv32F
				//	Lv16S
			}
			// else the image is added to the Training Data 
			else
			{
				//Xt32F
				//	Lt16S
			}
		}
	}

	FaceDetector::FaceDetector(const std::string &path_pos, const std::string &path_neg, const cv::Size &input_size,
		const int cell_size, const cv::Rect &crop_window, const int max_images, const double pos2neg_ratio) :
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
	void FaceDetector::load(MatVec &positiveImages, MatVec &negativeImages, const bool do_crop, const bool do_scale) {
		std::vector<std::string> positivesTraining;
		std::vector<std::string> positivesValidation;
		std::vector<std::string> negatives;

		//Read from the paths and load everything including the metadata paths into the PathVects
		cv::FileStorage fs(m_pos_path, cv::FileStorage::READ);
		fs["Training"] >> positivesTraining;
		fs["Validation"] >> positivesValidation;

		fs = cv::FileStorage(m_neg_path, cv::FileStorage::READ);
		fs["Files"] >> negatives;

		if ((do_crop == true) && (do_scale == true))
		{
			std::cout << "Loading positive training images..." << std::endl;
			//Positive training images
			for each (std::string path in positivesTraining) {
				cv::Mat image = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
				cv::resize(image(m_crop), image, m_model_size);

				positiveImages.push_back(image);
				m_img_fns_pos.push_back(path);
				m_img_fns_train.push_back(path);
			}
			std::cout << "Loading positive validation images..." << std::endl;
			//Positive validation images
			for each (std::string path in positivesValidation) {
				cv::Mat image = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
				cv::resize(image(m_crop), image, m_model_size);

				positiveImages.push_back(image);
				m_img_fns_pos.push_back(path);
				m_img_fns_validation.push_back(path);
			}
			std::cout << "Loading negative images..." << std::endl;
			for each (std::string path in negatives) {
				cv::Mat image = cv::imread(path, CV_LOAD_IMAGE_GRAYSCALE);
				//todo: crasht op croppen bij sommige plaatjes omdat ze te klein zijn (moeten we deze wel croppen)
				//cv::resize(image(m_crop), image, m_model_size);

				negativeImages.push_back(image);
				m_img_fns_neg.push_back(path);
				m_img_fns_neg_meta.push_back(path.replace(path.replace(23, 9, "Annotations").size() - 3, 3, "xml"));
			}
		}
	}
}