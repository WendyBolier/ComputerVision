/*
* FaceDetector.cpp
*
*  Created on : Jan 12, 2016
*      Authors : Erik and Wendy
*/

#include <iostream>

#include <boost/filesystem.hpp>

#include <opencv2/core/core.hpp>
#include <opencv2/ml/ml.hpp>
#include <opencv2/highgui/highgui.hpp>

#include "FaceDetector.h"
#include "MySVM.h"


/* Toggles */
//#define _FindBestC


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
	std::vector<int> FaceDetector::load(cv::Mat &trainingSamples, cv::Mat &validationSamples, const bool do_crop, const bool do_scale) {
		std::vector<std::string> positivesTraining, positivesValidation, negativesTraining, negativesValidation;

		//Read from the paths and load everything including the metadata paths into the PathVects
		cv::FileStorage fs(m_pos_path, cv::FileStorage::READ);
		fs["Training"] >> positivesTraining;
		fs["Validation"] >> positivesValidation;

		fs = cv::FileStorage(m_neg_path, cv::FileStorage::READ);
		fs["Training"] >> negativesTraining;
		fs["Validation"] >> negativesValidation;;

		if ((do_crop == true) && (do_scale == true)) {
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
			offsets.push_back(trainingSamples.rows);
			//Validation offset
			offsets.push_back(validationSamples.rows);

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

			std::cout << "Skipped " << tooSmallCounter << (tooSmallCounter == 1 ? " image because it is" : " images because they are") /*Yeah, I just did that*/ << " too small to crop..." << std::endl;
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
		return features;
	}

	void FaceDetector::svmFaces(const cv::Mat &trainingData, const cv::Mat &validationData, MySVM &svm, std::vector<int> offsets, SVMModel &model) {
		float bestC;
		double accuracyTraining, accuracyValidation, accuracyDifference = 100000;
		cv::Mat trainResults, validationResults;

		//Initialise the labels matrixes, 1 until the negative offset, -1 after the offset
		cv::Mat trainingLabels(trainingData.rows, 1, CV_32F), validationLabels(validationData.rows, 1, CV_32F);
		trainingLabels.setTo(-1);
		cv::Mat roi(trainingLabels(cv::Rect(0, 0, 1, offsets[0])));
		roi.setTo(1);

		validationLabels.setTo(-1);
		roi = cv::Mat(validationLabels(cv::Rect(0, 0, 1, offsets[1])));
		roi.setTo(1);

		//Initialise the parameters
		cv::SVMParams params;
		params.svm_type = cv::SVM::C_SVC;
		params.kernel_type = cv::SVM::POLY;
		params.degree = 1;
		params.term_crit = cv::TermCriteria(CV_TERMCRIT_ITER, 100000, 1e-6);

		std::cout << "Starting training... This might take a while..." << std::endl;
		
		//With differences of around 0.002, we found 0.008 to be the best value by running the code below
		bestC = 0.008;

		//Test different C values
#ifdef _FindBestC
		//for (params.C = 0.1; params.C < 1; params.C = params.C += 0.1) {
		for (params.C = 0.004; params.C < 0.011; params.C = params.C += 0.001) {
#else
		for (params.C = bestC; params.C == bestC; params.C++) {
#endif
			std::cout << "Training with C = " << params.C << "... ";
			svm.train(trainingData, trainingLabels, cv::Mat(), cv::Mat(), params);
			std::cout << "Training complete!" << std::endl;

			const int sv_count = svm.get_support_vector_count();
			const int sv_length = svm.get_var_count();

			CvSVMDecisionFunc *decision = svm.getDecisionFunc();
			model.weights = cv::Mat(sv_length, 1, CV_32F, 0.0);
			//Construct the weights Mat already transposed
			for (int i = 0; i < sv_count; i++) {
				const float *support_vector = svm.get_support_vector(i);
				const double weight = decision->alpha[i];
				for (int j = 0; j < sv_length; j++) {
					model.weights.at<float>(j, 0) += support_vector[j] * -weight;
				}
			}

			//Set up hyperplane
			const double bias = decision->rho;
			model.train_scores = (trainingData * model.weights) + bias;
			cv::compare(model.train_scores / cv::abs(model.train_scores), trainingLabels, trainResults, cv::CMP_EQ);
			double correctTraining = cv::sum(trainResults / 255)[0];
			accuracyTraining = correctTraining / trainingData.rows;
	
			//Save the support vectors (the number is off a bit due to float rounding)
			for (int i = 0; i < model.train_scores.rows; i++) {
				if (abs(model.train_scores.at<float>(i, 0)) <= 1) {
					model.support_vector_idx.emplace(i);
				}
			}
			
			model.validation_scores = (validationData * model.weights) + bias;
			cv::compare(model.validation_scores / cv::abs(model.validation_scores), validationLabels, validationResults, cv::CMP_EQ);
			double correctValidation = cv::sum(validationResults / 255)[0];
			accuracyValidation = correctValidation / validationData.rows;

			std::cout << "The training accuracy is " << accuracyTraining << " and the validation accuracy is " << accuracyValidation << std::endl;
			
			//Stop after the final training
			if (params.C == bestC) {
				break;
			}
			
			if (std::abs(accuracyTraining - accuracyValidation) < accuracyDifference) {
				bestC = params.C;
				accuracyDifference = std::abs(accuracyTraining - accuracyValidation);
			}

			//Do the final training once more with the bestC
			if (params.C == 1) {
				params.C = bestC;
			}
		}

		std::cout << "The best value for C is " << bestC << std::endl;

		//Visualise the weights vector
		cv::Mat reshapedWeights = model.weights.reshape(1, m_model_size.height);
		//cv::imshow("Display", (reshapedWeights * 1.6) + 0.6);
		{
			cv::Mat temp = ((reshapedWeights * 1.6) + 0.6) * 255;
			temp.convertTo(temp, CV_8U);
			cv::imwrite("FaceReconstruction.png", temp);
		}
		std::cout << std::endl << "The visualisation of W has been saved to the project folder..." << std::endl;

		//Save the filter engine
		int channels = 1; // Pixel model
		MatVec model_channels;
		cv::split(reshapedWeights, model_channels);
		int type = reshapedWeights.depth();
		int actualtype = reshapedWeights.type();

		for (size_t m = 0; m < model_channels.size(); m++) {
			auto channel_engine = cv::createLinearFilter(type, type, model_channels[m], cv::Point(-1, -1), 0, cv::BORDER_CONSTANT, -1, cv::Scalar(0, 0, 0, 0));
			model.engine.push_back(channel_engine);
		}
	}

	void FaceDetector::createPyramid(const float scaleFactor, cv::Mat &src, ImagePyramid &pyramid) {
		cv::Mat normalised = normalize(src);
		normalised = normalised.reshape(1, src.rows);
		float currentScaleFactor = scaleFactor;
		int depth = 1;

		//Keep making layers until the model doesn't fit in the image anymore; size of the model is multiplied to prevent senseless large models
		while (normalised.cols >= 5 * m_model_size.width * currentScaleFactor && normalised.rows >= 5 * m_model_size.height * currentScaleFactor) {
			Layer layer;
			cv::Mat scaled;
			cv::Size size((float)normalised.cols / currentScaleFactor, (float)normalised.rows / currentScaleFactor);
			//pyrDown limits the possible scales
			//cv::pyrDown(src, scaled, size);
			cv::resize(normalised, scaled, size);

			layer.factor = currentScaleFactor;
			layer.l = depth;
			layer.features = scaled;

			pyramid.push_back(layer);
			currentScaleFactor *= scaleFactor;
			depth++;
		}
	}

	void FaceDetector::convolve(SVMModel &model, MySVM &svm, Layer &pyramid_layer) {
		const auto roi = cv::Rect(0, 0, -1, -1); // convolve the full image
		const auto offset = cv::Point(0, 0);
		CvSVMDecisionFunc *decision = svm.getDecisionFunc();
		const double bias = decision->rho;
		pyramid_layer.pdf = cv::Mat(pyramid_layer.features.rows, pyramid_layer.features.cols, pyramid_layer.features.type(), 0.0);
		for (int c = 0; c < model.engine.size(); c++) {
			cv::Mat response;
			response.create(pyramid_layer.features.size(), pyramid_layer.features.type());
			int type = pyramid_layer.features.depth();
			model.engine[c]->apply(pyramid_layer.features, response, roi, offset, true);
			// Sum the responses in the PDF
			pyramid_layer.pdf += response;
		}
		// Add the bias to the PDF
		pyramid_layer.pdf += bias;
	}

	void FaceDetector::positionalContent(const Layer &pyramid_layer, const double threshold, CandidateVec &candidates) {
		for (int i = 0; i < pyramid_layer.pdf.rows; i++) {
			for (int j = 0; j < pyramid_layer.pdf.cols; j++) {
				if (pyramid_layer.pdf.at<float>(i, j) > threshold) {
					Candidate c;
					c.l = pyramid_layer.l;
					c.ftr_roi = cv::Rect(j - (m_model_size.width / 2), i - (m_model_size.height / 2), m_model_size.width, m_model_size.height);
					c.img_roi = cv::Rect(j * pyramid_layer.factor - (m_model_size.width * pyramid_layer.factor) / 2,
										 i * pyramid_layer.factor - (m_model_size.height * pyramid_layer.factor) / 2,
						                 m_model_size.width * pyramid_layer.factor, m_model_size.height * pyramid_layer.factor);
					c.score = pyramid_layer.pdf.at<float>(i, j);

					//Insert sorted by score
					int count = 0;
					while (candidates.size() > count && candidates[count].score > c.score) {
						count++;
					}
					candidates.insert(candidates.begin() + count, c);
				}
			}
		}
	}

	void FaceDetector::nonMaximaSuppression(const cv::Size &image_size, CandidateVec &candidates) {
		// We already sort the candidates on construction
		/*
		std::sort(candidates.begin(), candidates.end(), [](const Candidate &a, const Candidate &b) {
			return a.score > b.score;
		});
		*/

		cv::Mat scratch = cv::Mat::zeros(image_size, CV_8U);
		const cv::Rect bounds = cv::Rect(0, 0, 0, 0) + image_size;
		const double overlap = 0.5;
		unsigned int keep = 0;
		for (size_t n = 0; n < candidates.size(); n++) {
			cv::Rect box = candidates[n].img_roi & bounds;
			double scratchCount = 0;
			for (int i = box.y; i < box.y + box.height; i++) {
				for (int j = box.x; j < box.x + box.width; j++) {
					scratchCount += scratch.at<unsigned char>(i, j);
				}
			}
			// If not overlapping more than overlap, we keep this candidate
			if (scratchCount / (box.width * box.height) < overlap) {
				for (int i = box.y; i < box.y + box.height; i++) {
					for (int j = box.x; j < box.x + box.width; j++) {
						scratch.at<unsigned char>(i, j) = 1;
					}
				}
				//std::cout << box << " kept" << std::endl;
				keep++;
			}
			else {
				candidates.erase(candidates.begin() + n);
				//std::cout << box << " discarded" << std::endl;
				n--;
			}
		}
	}
}