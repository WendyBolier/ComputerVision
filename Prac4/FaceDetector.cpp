/*
* FaceDetector.cpp
*
*  Created on : Jan 12, 2016
*      Authors : Erik and Wendy
*/

#include "src\FaceDetector.h"

namespace nl_uu_science_gmt {


	FaceDetector::FaceDetector(
		&m_pos_path(0),
		&m_neg_path(0),
		&m_model_size(0))
	{

		

	/*	void prepare(const MatVec &pos_examples, const MatVec &neg_examples, const double factor,
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
					Xv32F
						Lv16S
				}
				// else the image is added to the Training Data 
				else
				{
					Xt32F
						Lt16S
				}
			}


		}*/




	}

	FaceDetector::~FaceDetector(

}