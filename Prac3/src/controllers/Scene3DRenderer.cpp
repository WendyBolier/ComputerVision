/*
 * Scene3DRenderer.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/imgproc/types_c.h>
#include <opencv2/core/mat.hpp>
#include <stddef.h>
#include <string>

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

	int numberOfClusters = 4;

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

	//TODO: als we geen centers hebben van het vorige frame, k-means om mee te beginnen (en de loop track op de grond clearen?)
	
	if ((centers.rows == 0) && (centers.cols == 0)) // of, als dit niet werkt: if(m_current_frame == 0) 
														// of een nieuwe variabele aanmaken: bool initialClusteringDone
														// (en die op true zetten als we de initial clustering gedaan hebben)
	{
		initialSpatialVoxelClustering();
	}
	
	//TODO: anders de nieuwe voxels goed labelen
	//TODO: cluster center bepalen (mean? of ook k-means?) en loop track tekenen (lijn van oude positie naar nieuwe positie)
	else
	{
		std::vector<Reconstructor::Voxel*> newVoxels = getNewVoxels();
		for (int i = 0; i < newVoxels.size(); i++)
		{
			Reconstructor::Voxel voxel = *newVoxels[i];
			float d1 = calculateDistance(voxel, Point(centers.at<float>(0, 0), centers.at <float>(0, 1)));
			float d2 = calculateDistance(voxel, Point(centers.at<float>(1, 0), centers.at <float>(1, 1)));
			float d3 = calculateDistance(voxel, Point(centers.at<float>(2, 0), centers.at <float>(2, 1)));
			float d4 = calculateDistance(voxel, Point(centers.at<float>(3, 0), centers.at <float>(3, 1)));

			if ((d1 < d2) && (d1 < d3) && (d1 < 4)) { /* add voxel to cluster 1*/ } 
	        else if ((d2 < d1) && (d2 < d3) && (d2 < d4)) { /* add voxel to cluster 2 */ } 
			else if ((d3 < d1) && (d3 < d2) && (d3 < d4)) { /* add voxel to cluster 3 */ }
			else if ((d4 < d1) && (d4 < d3) && (d4 < d2)) { /* add voxel to cluster 4 */ }
		}

		recluster();
		drawPaths();
	}

	previousVoxels = voxels;

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

	previousCenters = centers;
	kmeans(samples, numberOfClusters, labels, termCriteria, attempts, flags, centers);
}

/*
* Returns the new voxels that have appeared
*/
std::vector<Reconstructor::Voxel*> Scene3DRenderer::getNewVoxels()
{
	voxels = m_reconstructor.getVisibleVoxels();

	
	// TO DO: compare the previous voxels with the new visible voxels and return the different ones 
	
	for (int i = 0; i < max(voxels.size(), previousVoxels.size()); i++)
	{

	}

	return voxels;
}

/*
* Calculates the distance (in 2D space) between a voxel and a point 
*/
float Scene3DRenderer::calculateDistance(Reconstructor::Voxel v, Point p)
{
	return sqrt((v.x - p.x) * (v.x - p.x) + (v.y - p.y) * (v.y - p.y));
}

/*
* Draws the walking paths of the people on the floor
*/
void Scene3DRenderer::drawPaths()
{
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



} /* namespace nl_uu_science_gmt */
