/*
 * Reconstructor.cpp
 *
 *  Created on: Nov 15, 2013
 *      Author: coert
 */

#include "Reconstructor.h"

#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <cassert>
#include <iostream>

#include "../utilities/General.h"

using namespace std;
using namespace cv;

namespace nl_uu_science_gmt
{

/**
 * Constructor
 * Voxel reconstruction class
 */
Reconstructor::Reconstructor(
		const vector<Camera*> &cs) :
				m_cameras(cs),
				m_height(2048),
				m_step(32)
				//m_step(2048)
{
	for (size_t c = 0; c < m_cameras.size(); ++c)
	{
		if (m_plane_size.area() > 0)
			assert(m_plane_size.width == m_cameras[c]->getSize().width && m_plane_size.height == m_cameras[c]->getSize().height);
		else
			m_plane_size = m_cameras[c]->getSize();
	}

	const size_t edge = 2 * m_height;
	m_voxels_amount = (edge / m_step) * (edge / m_step) * (m_height / m_step);

	initialize();
}

/**
 * Deconstructor
 * Free the memory of the pointer vectors
 */
Reconstructor::~Reconstructor()
{
	for (size_t c = 0; c < m_corners.size(); ++c)
		delete m_corners.at(c);
	for (size_t v = 0; v < m_voxels.size(); ++v)
		delete m_voxels.at(v);
}

int Reconstructor::getVoxelIndex(int x, int y, int z) {
	const int xL = -m_height;
	const int xR = m_height;
	const int yL = -m_height;
	const int yR = m_height;
	const int zL = 0;
	const int zR = m_height;
	const int plane_y = (yR - yL) / m_step;
	const int plane_x = (xR - xL) / m_step;
	const int plane = plane_y * plane_x;
	const int zp = (z - zL) / m_step;
	const int yp = (y - yL) / m_step;
	const int xp = (x - xL) / m_step;

	return zp * plane + yp * plane_x + xp;  // The voxel's index
}

/**
 * Create some Look Up Tables
 * 	- LUT for the scene's box corners
 * 	- LUT with a map of the entire voxelspace: point-on-cam to voxels
 * 	- LUT with a map of the entire voxelspace: voxel to cam points-on-cam
 */
void Reconstructor::initialize()
{
	// Cube dimensions from [(-m_height, m_height), (-m_height, m_height), (0, m_height)]
	const int xL = -m_height;
	const int xR = m_height;
	const int yL = -m_height;
	const int yR = m_height;
	const int zL = 0;
	const int zR = m_height;
	const int plane_y = (yR - yL) / m_step;
	const int plane_x = (xR - xL) / m_step;
	const int plane = plane_y * plane_x;

	// Save the 8 volume corners
	// bottom
	m_corners.push_back(new Point3f((float) xL, (float) yL, (float) zL));
	m_corners.push_back(new Point3f((float) xL, (float) yR, (float) zL));
	m_corners.push_back(new Point3f((float) xR, (float) yR, (float) zL));
	m_corners.push_back(new Point3f((float) xR, (float) yL, (float) zL));

	// top
	m_corners.push_back(new Point3f((float) xL, (float) yL, (float) zR));
	m_corners.push_back(new Point3f((float) xL, (float) yR, (float) zR));
	m_corners.push_back(new Point3f((float) xR, (float) yR, (float) zR));
	m_corners.push_back(new Point3f((float) xR, (float) yL, (float) zR));

	// Acquire some memory for efficiency
	cout << "Initializing " << m_voxels_amount << " voxels ";
	m_voxels.resize(m_voxels_amount);

	int z;
	int pdone = 0;
#pragma omp parallel for schedule(auto) private(z) shared(pdone)
	for (z = zL; z < zR; z += m_step)
	{
		const int zp = (z - zL) / m_step;
		int done = cvRound((zp * plane / (double) m_voxels_amount) * 100.0);

#pragma omp critical
		if (done > pdone)
		{
			pdone = done;
			cout << done << "%..." << flush;
		}

		int y, x;
		for (y = yL; y < yR; y += m_step)
		{
			const int yp = (y - yL) / m_step;

			for (x = xL; x < xR; x += m_step)
			{
				const int xp = (x - xL) / m_step;

				// Create all voxels
				Voxel* voxel = new Voxel;
				voxel->x = x;
				voxel->y = y;
				voxel->z = z;
				voxel->visible = false;
				voxel->camera_projection = vector<Point>(m_cameras.size());
				voxel->valid_camera_projection = vector<int>(m_cameras.size(), 0);

				const int p = zp * plane + yp * plane_x + xp;  // The voxel's index

				for (size_t c = 0; c < m_cameras.size(); ++c)
				{
					Point point = m_cameras[c]->projectOnView(Point3f((float) x, (float) y, (float) z));

					// Save the pixel coordinates 'point' of the voxel projection on camera 'c'
					voxel->camera_projection[(int) c] = point;

					// If it's within the camera's FoV, flag the projection
					if (point.x >= 0 && point.x < m_plane_size.width && point.y >= 0 && point.y < m_plane_size.height)
						voxel->valid_camera_projection[(int) c] = 1;
				}

				//Writing voxel 'p' is not critical as it's unique (thread safe)
				m_voxels[p] = voxel;
			}
		}
	}

	cout << "done!" << endl;
}

/**
 * Count the amount of camera's each voxel in the space appears on,
 * if that amount equals the amount of cameras, add that voxel to the
 * visible_voxels vector
 */
void Reconstructor::update()
{
	m_visible_voxels.clear();
	std::vector<Voxel*> visible_voxels;

	int v;
#pragma omp parallel for schedule(auto) private(v) shared(visible_voxels)
	for (v = 0; v < (int)m_voxels_amount; ++v)
	{
		int camera_counter = 0;
		Voxel* voxel = m_voxels[v];

		for (size_t c = 0; c < m_cameras.size(); ++c)
		{
			if (voxel->valid_camera_projection[c])
			{
				const Point point = voxel->camera_projection[c];

				//If there's a white pixel on the foreground image at the projection point, add the camera
				if (m_cameras[c]->getForegroundImage().at<uchar>(point) == 255) ++camera_counter;
			}
		}

		// If the voxel is present on all cameras
		if (camera_counter == m_cameras.size())
		{
#pragma omp critical //push_back is critical
			visible_voxels.push_back(voxel);
			voxel->visible = true;
		}
	}
	
	// If the neighbours above and below, left and right or in front of behind a voxel are "on", then this voxel probably has to be "on" as well.
	for (int i = 0; i < (int)m_voxels_amount; i++)
	{
		Voxel* voxel = m_voxels[i];

		if (((voxel->z + m_step) >= 2048) || ((voxel->z - m_step) <= 0) || ((voxel->x - m_step) <= -2048) ||
			((voxel->x + m_step) >= 2048) || ((voxel->y - m_step) <= -2048) || ((voxel->y + m_step) >= 2048))
		{
			// we ignore the voxels on the borders
		}
		else
		{

			Voxel* voxelAbove = m_voxels[getVoxelIndex(voxel->x, voxel->y, voxel->z + m_step)];
			Voxel* voxelBelow = m_voxels[getVoxelIndex(voxel->x, voxel->y, voxel->z - m_step)];
			Voxel* voxelLeft = m_voxels[getVoxelIndex(voxel->x + m_step, voxel->y, voxel->z)];
			Voxel* voxelRight = m_voxels[getVoxelIndex(voxel->x - m_step, voxel->y, voxel->z)];
			Voxel* voxelFront = m_voxels[getVoxelIndex(voxel->x, voxel->y + m_step, voxel->z)];
			Voxel* voxelBehind = m_voxels[getVoxelIndex(voxel->x, voxel->y - m_step, voxel->z)];


			if (voxel->visible == false)
			{
				if ((voxelAbove->visible == true) && (voxelBelow->visible == true))
				{
					visible_voxels.push_back(voxel);
					voxel->visible = true;
				}
				else if ((voxelLeft->visible == true) && (voxelRight->visible == true))
				{
					visible_voxels.push_back(voxel);
					voxel->visible = true;
				}
				else if ((voxelFront->visible == true) && (voxelBehind->visible == true))
				{
					visible_voxels.push_back(voxel);
					voxel->visible = true;
				}
			}
		}
	}
	
	// If all neighbours around a voxel are "off", then this voxel probably has to be "off" as well.
	for (unsigned int j = 0; j < visible_voxels.size(); j++)
	{
		Voxel* voxel = visible_voxels[j];
		
		if (((voxel->z + m_step) >= 2048) || ((voxel->z - m_step) <= 0) || ((voxel->x - m_step) <= -2048) ||
			((voxel->x + m_step) >= 2048) || ((voxel->y - m_step) <= -2048) || ((voxel->y + m_step) >= 2048))
		{
			// we ignore the voxels on the borders
		}
		else
		{
			Voxel* voxelAbove = m_voxels[getVoxelIndex(voxel->x, voxel->y, voxel->z + m_step)];
			Voxel* voxelBelow = m_voxels[getVoxelIndex(voxel->x, voxel->y, voxel->z - m_step)];
			Voxel* voxelLeft = m_voxels[getVoxelIndex(voxel->x + m_step, voxel->y, voxel->z)];
			Voxel* voxelRight = m_voxels[getVoxelIndex(voxel->x - m_step, voxel->y, voxel->z)];
			Voxel* voxelFront = m_voxels[getVoxelIndex(voxel->x, voxel->y + m_step, voxel->z)];
			Voxel* voxelBehind = m_voxels[getVoxelIndex(voxel->x, voxel->y - m_step, voxel->z)];

			if ((voxelFront->visible == false) && (voxelBehind->visible == false) && (voxelLeft->visible == false) &&
				(voxelRight->visible == false) && (voxelFront->visible == false) && (voxelBehind->visible == false))
			{
				visible_voxels.erase(visible_voxels.begin() + j);
				voxel->visible = false;
			}
		}
	}
	m_visible_voxels.insert(m_visible_voxels.end(), visible_voxels.begin(), visible_voxels.end());
}

} /* namespace nl_uu_science_gmt */
