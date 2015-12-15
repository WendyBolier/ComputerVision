/*
 * VoxelReconstruction.h
 *
 *  Created on: Nov 13, 2013
 *      Author: coert
 */

#ifndef VOXELRECONSTRUCTION_H_
#define VOXELRECONSTRUCTION_H_

#include <string>
#include <vector>

#include "controllers/Camera.h"
#include "controllers/Scene3DRenderer.h"
#include "controllers/Glut.h"

namespace nl_uu_science_gmt
{

	class VoxelReconstruction
	{
		const std::string m_data_path;
		const int m_cam_views_amount;

		std::vector<Camera*> m_cam_views;

	public:
		VoxelReconstruction(const std::string &, const int);
		virtual ~VoxelReconstruction();

		static void showKeys();

		void initializeColorModels(Scene3DRenderer scene3d, Glut glut);
		void run(int, char**);	
	};

} /* namespace nl_uu_science_gmt */

#endif /* VOXELRECONSTRUCTION_H_ */
