/*
 * Scene3DRenderer.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/ml/ml.hpp>
#include <opencv2/core/mat.hpp>
#include <stddef.h>
#include <string>
#include <iostream>

#include "../utilities/General.h"
#include "Scene3DRenderer.h"

using namespace std;
using namespace cv;

namespace nl_uu_science_gmt
{

/**
 * Constructor
 * Scene properties class (mostly called by Glut)
 */
Scene3DRenderer::Scene3DRenderer(
		Reconstructor &r, const vector<Camera*> &cs) :
				m_reconstructor(r),
				m_cameras(cs),
				m_num(4),
				m_sphere_radius(1850)
{
	m_width = 640;
	m_height = 480;
	m_quit = false;
	m_paused = false;
	m_rotate = false;
	m_camera_view = true;
	m_show_volume = true;
	m_show_grd_flr = true;
	m_show_cam = true;
	m_show_org = true;
	m_show_arcball = false;
	m_show_info = true;
	m_fullscreen = false;

	// Read the checkerboard properties (XML)
	FileStorage fs;
	fs.open(m_cameras.front()->getDataPath() + ".." + string(PATH_SEP) + General::CBConfigFile, FileStorage::READ);
	if (fs.isOpened())
	{
		fs["CheckerBoardWidth"] >> m_board_size.width;
		fs["CheckerBoardHeight"] >> m_board_size.height;
		fs["CheckerBoardSquareSize"] >> m_square_side_len;
	}
	fs.release();

	m_current_camera = 0;
	m_previous_camera = 0;

	m_number_of_frames = m_cameras.front()->getFramesAmount();
	m_current_frame = 0;
	m_previous_frame = -1;

	const int H = 7;
	const int S = 17;
	const int V = 46;
	m_h_threshold = H;
	m_ph_threshold = H;
	m_s_threshold = S;
	m_ps_threshold = S;
	m_v_threshold = V;
	m_pv_threshold = V;

	numberOfClusters = 4;

	createTrackbar("Frame", VIDEO_WINDOW, &m_current_frame, m_number_of_frames - 2);
	createTrackbar("H", VIDEO_WINDOW, &m_h_threshold, 255);
	createTrackbar("S", VIDEO_WINDOW, &m_s_threshold, 255);
	createTrackbar("V", VIDEO_WINDOW, &m_v_threshold, 255);

	createFloorGrid();
	setTopView();
}

/**
 * Deconstructor
 * Free the memory of the floor_grid pointer vector
 */
Scene3DRenderer::~Scene3DRenderer()
{
	for (size_t f = 0; f < m_floor_grid.size(); ++f)
		for (size_t g = 0; g < m_floor_grid[f].size(); ++g)
			delete m_floor_grid[f][g];
}

/**
 * Process the current frame on each camera
 */
bool Scene3DRenderer::processFrame()
{
	for (size_t c = 0; c < m_cameras.size(); ++c)
	{
		if (m_current_frame == m_previous_frame + 1)
		{
			m_cameras[c]->advanceVideoFrame();
		}
		else if (m_current_frame != m_previous_frame)
		{
			m_cameras[c]->getVideoFrame(m_current_frame);
		}
		assert(m_cameras[c] != nullptr);
		processForeground(m_cameras[c]);
	}
	return true;
}

void Scene3DRenderer::initializeColorModels() {
	// Cluster the voxels
	int numberOfClusters = 4;
	Mat labels;
	Mat centers;
	TermCriteria termCriteria = TermCriteria(CV_TERMCRIT_ITER, 10000, 0.0001);
	int attempts = 3;
	int flags = KMEANS_PP_CENTERS; 
	std::vector<Reconstructor::Voxel*> voxels = getReconstructor().getVisibleVoxels();
	Mat samples(voxels.size(), 2, CV_32F);

	for (int x = 0; x < voxels.size(); x++) {
		samples.at<float>(x, 0) = voxels[x]->x;
		samples.at<float>(x, 1) = voxels[x]->y;
	}

	kmeans(samples, numberOfClusters, labels, termCriteria, attempts, flags, centers);

	Mat samplesPerson1(0, 1, CV_8UC3);
	Mat samplesPerson2(0, 1, CV_8UC3);
	Mat samplesPerson3(0, 1, CV_8UC3);
	Mat samplesPerson4(0, 1, CV_8UC3);

	// Find out which voxels belong to which person 

	// Create the samples for each person (from which the Gaussian mixture model will be estimated) 

	vector<Camera*> cameras = getCameras();
	vector<Mat> frames;
	//initialise frames of the right size
	frames.push_back(cameras.at(0)->getFrame());
	frames.push_back(cameras.at(1)->getFrame());
	frames.push_back(cameras.at(2)->getFrame());
	frames.push_back(cameras.at(3)->getFrame());

	//Make frames in HSV space
	cv::cvtColor(cameras.at(0)->getFrame(), frames[0], cv::COLOR_BGR2HSV);
	cv::cvtColor(cameras.at(1)->getFrame(), frames[1], cv::COLOR_BGR2HSV);
	cv::cvtColor(cameras.at(2)->getFrame(), frames[2], cv::COLOR_BGR2HSV);
	cv::cvtColor(cameras.at(3)->getFrame(), frames[3], cv::COLOR_BGR2HSV);

	std::vector<std::vector<Point>> usedPixelsPerCamera;
	for (int k = 0; k < cameras.size(); k++) {
		usedPixelsPerCamera.push_back(std::vector<Point>());
	}
	std::vector<std::vector<Reconstructor::Voxel>> voxelsPerPixel;

	//Get the colours for every pixel that are part of each person
	for (int i = 0; i < labels.rows; i++)
	{
		Reconstructor::Voxel* voxel = voxels[i];
		if (labels.at<int>(i, 0) == 0)
		{
			for (int k = 0; k < cameras.size(); k++) {
				Point point = voxel->camera_projection[k];

				//If the corresponding pixel has not yet been added to the model (for occlusion)
				if (std::find(usedPixelsPerCamera[k].begin(), usedPixelsPerCamera[k].end(), point) == usedPixelsPerCamera[k].end()) {
					samplesPerson1.push_back(frames[k].at<Vec3b>(point));
					usedPixelsPerCamera[k].push_back(point);
				}
			}
		}
		else if (labels.at<int>(i, 0) == 1)
		{
			for (int k = 0; k < cameras.size(); k++) {
				Point point = voxel->camera_projection[k];

				//If the corresponding pixel has not yet been added to the model (for occlusion)
				if (std::find(usedPixelsPerCamera[k].begin(), usedPixelsPerCamera[k].end(), point) == usedPixelsPerCamera[k].end()) {
					samplesPerson2.push_back(frames[k].at<Vec3b>(point));
					usedPixelsPerCamera[k].push_back(point);
				}
			}
		}
		else if (labels.at<int>(i, 0) == 2)
		{
			for (int k = 0; k < cameras.size(); k++) {
				Point point = voxel->camera_projection[k];

				//If the corresponding pixel has not yet been added to the model (for occlusion)
				if (std::find(usedPixelsPerCamera[k].begin(), usedPixelsPerCamera[k].end(), point) == usedPixelsPerCamera[k].end()) {
					samplesPerson3.push_back(frames[k].at<Vec3b>(point));
					usedPixelsPerCamera[k].push_back(point);
				}
			}
		}
		else if (labels.at<int>(i, 0) == 3)
		{
			for (int k = 0; k < cameras.size(); k++) {
				Point point = voxel->camera_projection[k];

				//If the corresponding pixel has not yet been added to the model (for occlusion)
				if (std::find(usedPixelsPerCamera[k].begin(), usedPixelsPerCamera[k].end(), point) == usedPixelsPerCamera[k].end()) {
					samplesPerson4.push_back(frames[k].at<Vec3b>(point));
					usedPixelsPerCamera[k].push_back(point);
				}
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

	//Make floats from the uchars, so EM can work with it
	Mat samplesFloat1, samplesFloat2, samplesFloat3, samplesFloat4;
	samplesPerson1.convertTo(samplesFloat1, CV_32F);
	samplesPerson2.convertTo(samplesFloat2, CV_32F);
	samplesPerson3.convertTo(samplesFloat3, CV_32F);
	samplesPerson4.convertTo(samplesFloat4, CV_32F);

	//Get a version without channels, but with 3 columns
	Mat samples1 = Mat(samplesPerson1.rows * samplesPerson1.cols, 3, CV_32F);
	Mat samples2 = Mat(samplesPerson2.rows * samplesPerson2.cols, 3, CV_32F);
	Mat samples3 = Mat(samplesPerson3.rows * samplesPerson3.cols, 3, CV_32F);
	Mat samples4 = Mat(samplesPerson4.rows * samplesPerson4.cols, 3, CV_32F);

	//Execute the EM training
	
	emPerson1 = cv::EM(numberOfColors, cv::EM::COV_MAT_DIAGONAL, termCriteria2);
	emPerson1.train(samples1, logLikelihoodsP1, labelsP1, probsP1);

	emPerson2 = cv::EM(numberOfColors, cv::EM::COV_MAT_DIAGONAL, termCriteria2);
	emPerson2.train(samples2, logLikelihoodsP2, labelsP2, probsP2);

	emPerson3 = cv::EM(numberOfColors, cv::EM::COV_MAT_DIAGONAL, termCriteria2);
	emPerson3.train(samples3, logLikelihoodsP3, labelsP3, probsP3);

	emPerson4 = cv::EM(numberOfColors, cv::EM::COV_MAT_DIAGONAL, termCriteria2);
	emPerson4.train(samples4, logLikelihoodsP4, labelsP4, probsP4);

	setCurrentFrame(0);
}

void Scene3DRenderer::updateClusters()
{
	if (getCurrentFrame() != getPreviousFrame())
	{
		initialSpatialVoxelClustering();
	}
	else
	{
		setLabels();
		recluster();
	}
	drawPaths();
	previousVoxels = voxels;
}

/**
 * Separate the background from the foreground
 * ie.: Create an 8 bit image where only the foreground of the scene is white (255)
 */
void Scene3DRenderer::processForeground(
		Camera* camera)
{
	assert(!camera->getFrame().empty());
	Mat hsv_image;
	cvtColor(camera->getFrame(), hsv_image, CV_BGR2HSV);  // from BGR to HSV color space

	vector<Mat> channels;
	split(hsv_image, channels);  // Split the HSV-channels for further analysis

	// Background subtraction H
	Mat tmp, foreground, background;
	absdiff(channels[0], camera->getBgHsvChannels().at(0), tmp);
	//threshold(tmp, foreground, m_h_threshold, 255, CV_THRESH_BINARY);
	threshold(tmp, foreground, 0, 255, THRESH_BINARY | THRESH_OTSU);
	
	// Background subtraction S
	absdiff(channels[1], camera->getBgHsvChannels().at(1), tmp);
	//threshold(tmp, background, m_s_threshold, 255, CV_THRESH_BINARY);
	threshold(tmp, background, 0, 255, THRESH_BINARY | THRESH_OTSU);
	bitwise_or(foreground, background, foreground);
	
	// Background subtraction V
	absdiff(channels[2], camera->getBgHsvChannels().at(2), tmp);
	//threshold(tmp, background, m_v_threshold, 255, CV_THRESH_BINARY);
	threshold(tmp, background, 0, 255, CV_THRESH_BINARY | THRESH_OTSU);
	bitwise_or(foreground, background, foreground);

	Mat kernel = kernel.zeros(3, 3, CV_8U);
	kernel.at<char>(0, 1) = 1;
	kernel.at<char>(1, 2) = 1;
	kernel.at<char>(1, 2) = 1;
	kernel.at<char>(2, 1) = 1;

	if (!useGraphCut) {
		erode(foreground, foreground, kernel, Point(-1, -1), 1);
		dilate(foreground, foreground, kernel, Point(-1, -1), 1);
	}
	else {
		// Our Graph Cut implementation - ultimately dropped
		dilate(foreground, foreground, kernel);
		erode(foreground, foreground, kernel, Point(-1, -1), 10);

		Mat bgModel, fgModel, mask(foreground.rows, foreground.cols, CV_8U);

		int xmin = 10000, xmax = 0, ymin = 10000, ymax = 0;
		for (int i = 0; i < mask.rows; i++){
			//skip the last 10 pixels, because there are falsely positive columns of pixels there
			for (int j = 0; j < mask.cols - 10; j++){
				if (foreground.at<uchar>(i, j) == 255) {
					mask.at<uchar>(i, j) = GC_FGD;
					xmin = min(xmin, j);
					xmax = max(xmax, j);
					ymin = min(ymin, i);
					ymax = max(ymax, i);
				}
				else {
					mask.at<uchar>(i, j) = GC_PR_BGD;
				}
			}
		}

		grabCut(camera->getFrame(), mask, Rect(xmin - 20, ymin - 20, xmax - xmin + 20, ymax - ymin + 20), bgModel, fgModel, 3, GC_INIT_WITH_RECT);

		for (int i = 0; i < mask.rows; i++) {
			for (int j = 0; j < mask.cols; j++) {
				uchar val = mask.at<uchar>(i, j);
				if (val == GC_FGD | val == GC_PR_FGD) {
					foreground.at<uchar>(i, j) = 255;
				}
				else {
					foreground.at<uchar>(i, j) = 0;
				}
			}
		}
	}
	camera->setForegroundImage(foreground);
}

/**
 * Set currently visible camera to the given camera id
 */
void Scene3DRenderer::setCamera(
		int camera)
{
	m_camera_view = true;

	if (m_current_camera != camera)
	{
		m_previous_camera = m_current_camera;
		m_current_camera = camera;
		m_arcball_eye.x = m_cameras[camera]->getCameraPlane()[0].x;
		m_arcball_eye.y = m_cameras[camera]->getCameraPlane()[0].y;
		m_arcball_eye.z = m_cameras[camera]->getCameraPlane()[0].z;
		m_arcball_up.x = 0.0f;
		m_arcball_up.y = 0.0f;
		m_arcball_up.z = 1.0f;
	}
}

/**
 * Set the 3D scene to bird's eye view
 */
void Scene3DRenderer::setTopView()
{
	m_camera_view = false;
	if (m_current_camera != -1)
		m_previous_camera = m_current_camera;
	m_current_camera = -1;

	m_arcball_eye = vec(0.0f, 0.0f, 10000.0f);
	m_arcball_centre = vec(0.0f, 0.0f, 0.0f);
	m_arcball_up = vec(0.0f, 1.0f, 0.0f);
}

/**
 * Create a LUT for the floor grid
 */
void Scene3DRenderer::createFloorGrid()
{
	const int size = m_reconstructor.getSize() / m_num;
	const int z_offset = 3;

	// edge 1
	vector<Point3i*> edge1;
	for (int y = -size * m_num; y <= size * m_num; y += size)
		edge1.push_back(new Point3i(-size * m_num, y, z_offset));

	// edge 2
	vector<Point3i*> edge2;
	for (int x = -size * m_num; x <= size * m_num; x += size)
		edge2.push_back(new Point3i(x, size * m_num, z_offset));

	// edge 3
	vector<Point3i*> edge3;
	for (int y = -size * m_num; y <= size * m_num; y += size)
		edge3.push_back(new Point3i(size * m_num, y, z_offset));

	// edge 4
	vector<Point3i*> edge4;
	for (int x = -size * m_num; x <= size * m_num; x += size)
		edge4.push_back(new Point3i(x, -size * m_num, z_offset));

	m_floor_grid.push_back(edge1);
	m_floor_grid.push_back(edge2);
	m_floor_grid.push_back(edge3);
	m_floor_grid.push_back(edge4);
}

/**
* Do the initial spatial voxel clustering with KMeans
*/
void Scene3DRenderer::initialSpatialVoxelClustering()
{
	TermCriteria termCriteria = TermCriteria(CV_TERMCRIT_ITER, 10000, 0.0001);
	int attempts = 3;
	int flags = KMEANS_PP_CENTERS;
	voxels = m_reconstructor.getVisibleVoxels();
	samples = Mat(voxels.size(), 2, CV_32F);

	for (int x = 0; x < voxels.size(); x++) {
		samples.at<float>(x, 0) = voxels[x]->x;
		samples.at<float>(x, 1) = voxels[x]->y;
	}

	kmeans(samples, numberOfClusters, labels, termCriteria, attempts, flags, centers);

	Mat samplesPerson1(0, 1, CV_8UC3);
	Mat samplesPerson2(0, 1, CV_8UC3);
	Mat samplesPerson3(0, 1, CV_8UC3);
	Mat samplesPerson4(0, 1, CV_8UC3);

	vector<Camera*> cameras = getCameras();
	vector<Mat> frames;
	//initialise frames of the right size
	frames.push_back(cameras.at(0)->getFrame());
	frames.push_back(cameras.at(1)->getFrame());
	frames.push_back(cameras.at(2)->getFrame());
	frames.push_back(cameras.at(3)->getFrame());

	//Make frames in HSV space
	cv::cvtColor(cameras.at(0)->getFrame(), frames[0], cv::COLOR_BGR2HSV);
	cv::cvtColor(cameras.at(1)->getFrame(), frames[1], cv::COLOR_BGR2HSV);
	cv::cvtColor(cameras.at(2)->getFrame(), frames[2], cv::COLOR_BGR2HSV);
	cv::cvtColor(cameras.at(3)->getFrame(), frames[3], cv::COLOR_BGR2HSV);

	std::vector<std::vector<Point>> usedPixelsPerCamera;
	for (int k = 0; k < cameras.size(); k++) {
		usedPixelsPerCamera.push_back(std::vector<Point>());
	}
	std::vector<std::vector<Reconstructor::Voxel>> voxelsPerPixel;

	//Get the colours for every pixel that are part of each person
	for (int i = 0; i < labels.rows; i++)
	{
		Reconstructor::Voxel* voxel = voxels[i];
		if (labels.at<int>(i, 0) == 0)
		{
			for (int k = 0; k < cameras.size(); k++) {
				Point point = voxel->camera_projection[k];

				//If the corresponding pixel has not yet been added to the model (for occlusion)
				if (std::find(usedPixelsPerCamera[k].begin(), usedPixelsPerCamera[k].end(), point) == usedPixelsPerCamera[k].end()) {
					samplesPerson1.push_back(frames[k].at<Vec3b>(point));
					usedPixelsPerCamera[k].push_back(point);
				}
			}
		}
		else if (labels.at<int>(i, 0) == 1)
		{
			for (int k = 0; k < cameras.size(); k++) {
				Point point = voxel->camera_projection[k];

				//If the corresponding pixel has not yet been added to the model (for occlusion)
				if (std::find(usedPixelsPerCamera[k].begin(), usedPixelsPerCamera[k].end(), point) == usedPixelsPerCamera[k].end()) {
					samplesPerson2.push_back(frames[k].at<Vec3b>(point));
					usedPixelsPerCamera[k].push_back(point);
				}
			}
		}
		else if (labels.at<int>(i, 0) == 2)
		{
			for (int k = 0; k < cameras.size(); k++) {
				Point point = voxel->camera_projection[k];

				//If the corresponding pixel has not yet been added to the model (for occlusion)
				if (std::find(usedPixelsPerCamera[k].begin(), usedPixelsPerCamera[k].end(), point) == usedPixelsPerCamera[k].end()) {
					samplesPerson3.push_back(frames[k].at<Vec3b>(point));
					usedPixelsPerCamera[k].push_back(point);
				}
			}
		}
		else if (labels.at<int>(i, 0) == 3)
		{
			for (int k = 0; k < cameras.size(); k++) {
				Point point = voxel->camera_projection[k];

				//If the corresponding pixel has not yet been added to the model (for occlusion)
				if (std::find(usedPixelsPerCamera[k].begin(), usedPixelsPerCamera[k].end(), point) == usedPixelsPerCamera[k].end()) {
					samplesPerson4.push_back(frames[k].at<Vec3b>(point));
					usedPixelsPerCamera[k].push_back(point);
				}
			}
		}
	}

	//Make floats from the uchars, so EM can work with it
	Mat samplesFloat1, samplesFloat2, samplesFloat3, samplesFloat4;
	samplesPerson1.convertTo(samplesFloat1, CV_32F);
	samplesPerson2.convertTo(samplesFloat2, CV_32F);
	samplesPerson3.convertTo(samplesFloat3, CV_32F);
	samplesPerson4.convertTo(samplesFloat4, CV_32F);

	//Get a version without channels, but with 3 columns
	Mat samples1 = Mat(samplesPerson1.rows * samplesPerson1.cols, 3, CV_32F);
	Mat samples2 = Mat(samplesPerson2.rows * samplesPerson2.cols, 3, CV_32F);
	Mat samples3 = Mat(samplesPerson3.rows * samplesPerson3.cols, 3, CV_32F);
	Mat samples4 = Mat(samplesPerson4.rows * samplesPerson4.cols, 3, CV_32F);

	Vec2d prediction1, prediction2, prediction3, prediction4;

	prediction1 = emPerson1.predict(samples1);
	prediction2 = emPerson2.predict(samples2);
	prediction3 = emPerson3.predict(samples3);
	prediction4 = emPerson4.predict(samples4);
}

/**
* Re-­cluster the voxels based on the current labeling state.
*/
void Scene3DRenderer::recluster()
{
	TermCriteria termCriteria = TermCriteria(CV_TERMCRIT_ITER, 100, 0.01);
	int attempts = 1;
	int flags = KMEANS_USE_INITIAL_LABELS;
	voxels = m_reconstructor.getVisibleVoxels();

	samples = Mat(voxels.size(), 2, CV_32F);
	for (int x = 0; x < voxels.size(); x++) {
		samples.at<float>(x, 0) = voxels[x]->x;
		samples.at<float>(x, 1) = voxels[x]->y;
	}

	labels = Mat(voxels.size(), labels.cols, labels.type());
	for (int x = 0; x < voxels.size(); x++) {
		labels.at<int>(x, 0) = voxels[x]->label;
	}

	previousCenters = centers;
	kmeans(samples, numberOfClusters, labels, termCriteria, attempts, flags, centers);
}

/*
* Calculates the distance (in 2D space) between a voxel and a point 
*/
float Scene3DRenderer::calculateDistance(Reconstructor::Voxel v, Point2f p)
{
	return sqrt((v.x - p.x) * (v.x - p.x) + (v.y - p.y) * (v.y - p.y));
}

/*
* Draws the walking paths of the people on the floor
*/
void Scene3DRenderer::drawPaths()
{
	//If there are no previousCenters
	if (previousCenters.dims < 2) {
		return;
	}
	Point c1p1 = (previousCenters.at<float>(0, 0), previousCenters.at<float>(0, 1));
	Point c1p2 = (centers.at<float>(0, 0), centers.at<float>(0, 1));
	Point c2p1 = (previousCenters.at<float>(1, 0), previousCenters.at<float>(1, 1));
	Point c2p2 = (centers.at<float>(1, 0), centers.at<float>(1, 1));
	Point c3p1 = (previousCenters.at<float>(2, 0), previousCenters.at<float>(2, 1));
	Point c3p2 = (centers.at<float>(2, 0), centers.at<float>(2, 1));
	Point c4p1 = (previousCenters.at<float>(3, 0), previousCenters.at<float>(3, 1));
	Point c4p2 = (centers.at<float>(3, 0), centers.at<float>(3, 1));

	line(paths, c1p1, c1p2, Scalar(255, 20, 147), 1, 8, 0);
	line(paths, c2p1, c2p2, Scalar(255, 0, 0), 1, 8, 0);
	line(paths, c3p1, c3p2, Scalar(0, 255, 0), 1, 8, 0);
	line(paths, c4p1, c4p2, Scalar(0, 0, 255), 1, 8, 0);
}

/*
* Adds and removes labels from voxels
*/
void Scene3DRenderer::setLabels()
{
	std::vector<Reconstructor::Voxel*> voxels = m_reconstructor.getVoxels();
	// Delete the labels from voxels that are not visible anymore 
	for (int k = 0; k < voxels.size(); k++)
	{
		Reconstructor::Voxel* v = voxels[k];
		if (!v->visible)
		{
			v->label = -1;
		}
		else if (v->label == -1) {
			float dist1 = calculateDistance(*v, Point(centers.at<float>(0, 0), centers.at<float>(0, 1)));
			float dist2 = calculateDistance(*v, Point(centers.at<float>(1, 0), centers.at<float>(1, 1)));
			float dist3 = calculateDistance(*v, Point(centers.at<float>(2, 0), centers.at<float>(2, 1)));
			float dist4 = calculateDistance(*v, Point(centers.at<float>(3, 0), centers.at<float>(3, 1)));
			
			float min = min(dist1, min(dist2, min(dist3, dist4)));
			if (min == dist1)
				v->label = 0;
			else if (min == dist2)
				v->label = 1;
			else if (min == dist3)
				v->label = 2;
			else if (min == dist4)
				v->label = 3;
		}
	}
}


} /* namespace nl_uu_science_gmt */
