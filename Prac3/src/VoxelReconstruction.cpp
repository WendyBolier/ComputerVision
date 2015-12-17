/*
 * VoxelReconstruction.cpp
 *
 *  Created on: Nov 13, 2013
 *      Author: coert
 */

#include "VoxelReconstruction.h"

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/highgui/highgui_c.h>
#include <opencv2/ml/ml.hpp>
#include <stddef.h>
#include <cassert>
#include <iostream>
#include <sstream>

#include "controllers/Glut.h"
#include "controllers/Reconstructor.h"
#include "controllers/Scene3DRenderer.h"
#include "utilities/General.h"
#include "CalibrateCameras.h"

using namespace nl_uu_science_gmt;
using namespace std;
using namespace cv;

namespace nl_uu_science_gmt
{

/**
 * Main constructor, initialized all cameras
 */
VoxelReconstruction::VoxelReconstruction(const string &dp, const int cva) :
		m_data_path(dp), m_cam_views_amount(cva)
{
	const string cam_path = m_data_path + "cam";

	for (int v = 0; v < m_cam_views_amount; ++v)
	{
		stringstream full_path;
		full_path << cam_path << (v + 1) << PATH_SEP;

		/*
		 * Assert that there's a background image or video file and \
		 * that there's a video file
		 */
		std::cout << full_path.str() << General::BackgroundImageFile << std::endl;
		std::cout << full_path.str() << General::VideoFile << std::endl;
		assert(
			General::fexists(full_path.str() + General::BackgroundImageFile)
			&&
			General::fexists(full_path.str() + General::VideoFile)
		);

		if (!General::fexists(full_path.str() + General::IntrinsicsFile))
			calibrateCamera(v);

		/*
		 * Assert that if there's no config.xml file, there's an intrinsics file and
		 * a checkerboard video to create the extrinsics from
		 */
		assert(
			(!General::fexists(full_path.str() + General::ConfigFile) ?
				General::fexists(full_path.str() + General::IntrinsicsFile) &&
					General::fexists(full_path.str() + General::CheckerboadVideo)
			 : true)
		);

		m_cam_views.push_back(new Camera(full_path.str(), General::ConfigFile, v));
	}

	
}

/**
 * Main destructor, cleans up pointer vector memory of the cameras
 */
VoxelReconstruction::~VoxelReconstruction()
{
	for (size_t v = 0; v < m_cam_views.size(); ++v)
		delete m_cam_views[v];
}

/**
 * What you can hit
 */
void VoxelReconstruction::showKeys()
{
	cout << "VoxelReconstruction v" << VERSION << endl << endl;
	cout << "Use these keys:" << endl;
	cout << "q       : Quit" << endl;
	cout << "p       : Pause" << endl;
	cout << "b       : Frame back" << endl;
	cout << "n       : Next frame" << endl;
	cout << "r       : Rotate voxel space" << endl;
	cout << "s       : Show/hide arcball wire sphere (Linux only)" << endl;
	cout << "v       : Show/hide voxel space box" << endl;
	cout << "g       : Show/hide ground plane" << endl;
	cout << "c       : Show/hide cameras" << endl;
	cout << "i       : Show/hide camera numbers (Linux only)" << endl;
	cout << "o       : Show/hide origin" << endl;
	cout << "t       : Top view" << endl;
	cout << "1,2,3,4 : Switch camera #" << endl << endl;
	cout << "Zoom with the scrollwheel while on the 3D scene" << endl;
	cout << "Rotate the 3D scene with left click+drag" << endl << endl;
}

void VoxelReconstruction::initializeColorModels(Scene3DRenderer scene3d, Glut glut) {
	scene3d.setCurrentFrame(594);

	glut.update(0);

	// Cluster the voxels
	int numberOfClusters = 4;
	Mat labels;
	Mat centers;
	TermCriteria termCriteria = TermCriteria(CV_TERMCRIT_ITER, 10000, 0.0001);
	int attempts = 3;
	int flags = KMEANS_PP_CENTERS; // see to do
	std::vector<Reconstructor::Voxel*, std::allocator<Reconstructor::Voxel*>> voxels = scene3d.getReconstructor().getVisibleVoxels();
	Mat samples(voxels.size(), 2, CV_32F);

	for (int x = 0; x < voxels.size(); x++) {
		samples.at<float>(x, 0) = voxels[x]->x;
		samples.at<float>(x, 1) = voxels[x]->y;
	}

	kmeans(samples, numberOfClusters, labels, termCriteria, attempts, flags, centers);

	/* To do: decide which flag to use

	KMEANS_RANDOM_CENTERS Select random initial centers in each attempt.
	KMEANS_PP_CENTERS Use kmeans++ center initialization by Arthur and Vassilvitskii [Arthur2007].
	KMEANS_USE_INITIAL_LABELS During the first (and possibly the only) attempt, use the user-supplied labels instead of computing them from the initial centers. For the second and further attempts, use the random or semi-random centers. Use one of KMEANS_*_CENTERS flag to specify the exact method.

	*/

	std::vector<Reconstructor::Voxel*> voxelsPerson1, voxelsPerson2, voxelsPerson3, voxelsPerson4;

	Mat samplesPerson1(0, 2, CV_32S);
	Mat samplesPerson2(0, 2, CV_32S);
	Mat samplesPerson3(0, 2, CV_32S);
	Mat samplesPerson4(0, 2, CV_32S);

	// Find out which voxels belong to which person 

	// Create the samples for each person (from which the Gaussian mixture model will be estimated) 

	for (int i = 0; i < labels.rows; i++)
	{
		Reconstructor::Voxel* voxel = voxels[i];
		if (labels.at<int>(i, 0) == 0)
		{
			voxelsPerson1.push_back(voxel);
			for (int k = 0; k < scene3d.getCameras().size(); k++)
			{
				Mat temp(1, 2, CV_32S);
				Point p = voxel->camera_projection[k];
				temp.at<int>(0, 0) = p.x;
				temp.at<int>(0, 1) = p.y;
				samplesPerson1.push_back(temp);
			}
		}
		else if (labels.at<int>(i, 0) == 1)
		{
			voxelsPerson2.push_back(voxel);
			for (int k = 0; k < scene3d.getCameras().size(); k++)
			{
				Mat temp(1, 2, CV_32S);
				Point p = voxel->camera_projection[k];
				temp.at<int>(0, 0) = p.x;
				temp.at<int>(0, 1) = p.y;
				samplesPerson2.push_back(temp);
			}
		}
		else if (labels.at<int>(i, 0) == 2)
		{
			voxelsPerson3.push_back(voxel);
			for (int k = 0; k < scene3d.getCameras().size(); k++)
			{
				Mat temp(1, 2, CV_32S);
				Point p = voxel->camera_projection[k];
				temp.at<int>(0, 0) = p.x;
				temp.at<int>(0, 1) = p.y;
				samplesPerson3.push_back(temp);
			}
		}
		else if (labels.at<int>(i, 0) == 3)
		{
			voxelsPerson4.push_back(voxel);
			for (int k = 0; k < scene3d.getCameras().size(); k++)
			{
				Mat temp(1, 2, CV_32S);
				Point p = voxel->camera_projection[k];
				temp.at<int>(0, 0) = p.x;
				temp.at<int>(0, 1) = p.y;
				samplesPerson4.push_back(temp);
			}
		}
	}

	int numberOfColors = 3;
	const TermCriteria termCriteria2 = TermCriteria(TermCriteria::COUNT + TermCriteria::EPS, 100, FLT_EPSILON);

	Mat logLikelihoodsP1(samplesPerson1.rows, 1, CV_64FC1);
	Mat logLikelihoodsP2(samplesPerson2.rows, 1, CV_64FC1);
	Mat logLikelihoodsP3(samplesPerson3.rows, 1, CV_64FC1);
	Mat logLikelihoodsP4(samplesPerson4.rows, 1, CV_64FC1);

	Mat labelsP1(samplesPerson1.rows, 1, CV_32SC1);
	Mat labelsP2(samplesPerson2.rows, 1, CV_32SC1);
	Mat labelsP3(samplesPerson3.rows, 1, CV_32SC1);
	Mat labelsP4(samplesPerson4.rows, 1, CV_32SC1);

	Mat probsP1(samplesPerson1.rows, numberOfColors, CV_64FC1);
	Mat probsP2(samplesPerson2.rows, numberOfColors, CV_64FC1);
	Mat probsP3(samplesPerson3.rows, numberOfColors, CV_64FC1);
	Mat probsP4(samplesPerson4.rows, numberOfColors, CV_64FC1);

	cv::EM emPerson1, emPerson2, emPerson3, emPerson4;
	emPerson1 = cv::EM(numberOfColors, cv::EM::COV_MAT_DIAGONAL, termCriteria2);
	cout << emPerson1.train(samplesPerson1, logLikelihoodsP1, labelsP1, probsP1);

	emPerson2 = cv::EM(numberOfColors, cv::EM::COV_MAT_DIAGONAL, termCriteria2);
	cout << emPerson2.train(samplesPerson2, logLikelihoodsP2, labelsP2, probsP2);

	emPerson3 = cv::EM(numberOfColors, cv::EM::COV_MAT_DIAGONAL, termCriteria2);
	cout << emPerson3.train(samplesPerson3, logLikelihoodsP3, labelsP3, probsP3);

	emPerson4 = cv::EM(numberOfColors, cv::EM::COV_MAT_DIAGONAL, termCriteria2);
	cout << emPerson4.train(samplesPerson4, logLikelihoodsP4, labelsP4, probsP4);

	scene3d.setCurrentFrame(0);
}

/**
 * - If the xml-file with camera intrinsics, extrinsics and distortion is missing,
 *   create it from the checkerboard video and the measured camera intrinsics
 * - After that initialize the scene rendering classes
 * - Run it!
 */
void VoxelReconstruction::run(int argc, char** argv)
{
	for (int v = 0; v < m_cam_views_amount; ++v)
	{
		bool has_cam = Camera::detExtrinsics(m_cam_views[v]->getDataPath(), General::CheckerboadVideo,
				General::IntrinsicsFile, m_cam_views[v]->getCamPropertiesFile());
		assert(has_cam);
		if (has_cam) has_cam = m_cam_views[v]->initialize();
		assert(has_cam);
	}

	destroyAllWindows();
	namedWindow(VIDEO_WINDOW, CV_WINDOW_KEEPRATIO);

	Reconstructor reconstructor(m_cam_views);
	Scene3DRenderer scene3d(reconstructor, m_cam_views);
	Glut glut(scene3d);

#ifdef __linux__
	glut.initializeLinux(SCENE_WINDOW.c_str(), argc, argv);
	initializeColorModels(scene3d);
	glut.glutMainLoop();
#elif defined _WIN32
	glut.initializeWindows(SCENE_WINDOW.c_str());
	initializeColorModels(scene3d, glut);
	glut.mainLoopWindows();
#endif
}

} /* namespace nl_uu_science_gmt */
