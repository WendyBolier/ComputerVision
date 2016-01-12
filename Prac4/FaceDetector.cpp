/*
* FaceDetector.cpp
*
*  Created on : Jan 12, 2016
*      Authors : Erik and Wendy
*/

#include "src\FaceDetector.h"

namespace nl_uu_science_gmt
{

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
}