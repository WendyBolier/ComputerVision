/*
 * FeatureHOG.cpp
 *
 *  Created on: Feb 15, 2013
 *      Author: coert
 */

#include "FeatureHOG.h"

#include <iostream>
#ifdef _OPENMP
#include <omp.h>
#endif
#include <opencv2/core/core_c.h>
#include <opencv2/core/mat.hpp>
#include <opencv2/core/operations.hpp>
#include <opencv2/core/types_c.h>
#include <opencv2/imgproc/imgproc.hpp>
#include <stddef.h>
#include <algorithm>
#define _USE_MATH_DEFINES // for C++
#include <math.h>
#include <limits>

namespace nl_uu_science_gmt
{

FeatureHOG::FeatureHOG() :
				ci_padx(0),
				ci_pady(0),
				ci_bins(9),
				ci_lbp_bins(0),
				ct_trunc(0.2),
				ct_texture(0.2357),
				ct_step(0),
				cb_texture_features(true),
				ci_depth(ci_bins * 3 + 4) // (ci_bins [contrast_insensitive]) + (2 * ci_bins [contrast_sensitive]) + (4 [texture_features])
{
	int dx, dy;
	for (dy = -255; dy <= 255; ++dy)
	{
		for (dx = -255; dx <= 255; ++dx)
		{
			// Magnitude in the range [0, 1]
			const auto magnitude = std::sqrt((double) dx * dx + dy * dy) / 255.0;

			// Angle in the range [-pi, pi]
			auto angle = atan2(static_cast<double>(dy), static_cast<double>(dx));

			// Convert it to the range [BINS, BINS * 3]
			angle = angle * (ci_bins / M_PI) + ci_bins * 2;

			// Convert it to the range [0, 18]
			if (angle >= ci_bins * 2)
				angle -= ci_bins * 2;

			// Bilinear interpolation
			const auto bin0 = (int) angle;
			const auto bin1 = (int) (bin0 < ci_bins * 2 - 1) ? (bin0 + 1) : 0;
			const auto alpha = angle - bin0;

			u_qtz_bins[dy + 255][dx + 255][0] = bin0;
			u_qtz_bins[dy + 255][dx + 255][1] = bin1;
			d_qtz_mags[dy + 255][dx + 255][0] = (float) magnitude * (1.0 - alpha);
			d_qtz_mags[dy + 255][dx + 255][1] = (float) magnitude * alpha;
		}
	}
}

FeatureHOG::~FeatureHOG()
{
}

// *************************************** Felzenszwalb HOG ***********************************************

/**
 * Non-member function to statically compute HOG descriptor from an image the 'Felzenszwalb' way
 *
 * Example: FeatureHOG<float>::compute(image, features);
 *
 * cv::Mat features;
 * An interleaved matrix containing the HOG cell data which consists of cells with bins. The histogram bins
 * are interleaved with the cell columns:
 * Mat.height: "the vertical amount of cells"
 * Mat.width:  "the horizontal amount of cells x FeatureHOG::DEPTH"
 * This interleaves DEPTH slices  (= amount of bins per cell) into the X columns,
 * so HOG_cell_width = Mat.width / FeatureHOG::DEPTH
 *
 * Ie.:
 * Y0X0S0, Y0X0S1, Y0X0S2, Y0X0S3, ..., Y0X0S31, Y0X1S0, ..., Y0X19S1, Y0X19S2, Y0X19S3, ..., Y0X19S31
 * Y1X0S0, Y1X0S1, Y1X0S2, Y1X0S3, ..., Y1X0S31, Y1X1S0, ..., Y1X19S1, Y1X19S2, Y1X19S3, ..., Y1X19S31
 * Y2X0S0, Y2X0S1, Y2X0S2, Y2X0S3, ..., Y2X0S31, Y2X1S0, ..., Y2X19S1, Y2X19S2, Y2X19S3, ..., Y2X19S31
 * .
 * .
 *
 * FeatureHOG::DEPTH = The depth of the matrix, with each slice containing the:
 * contrast sensitive, contrast insensitive, texture energy and occlusion feature bins
 *
 * Use cv::split and cv::merge to get access to each slice for convolution. Eg.:
 *
 * cv::Mat features;
 * FeatureHOG<float>::compute(image, features);
 * std::vector<cv::Mat> feature_slices;
 * cv::split(features.reshape(getDepth(), feature_slices); // Reshape to getDepth() channels, and split into vector of channel slices
 * --operate on feature slices--
 * cv::merge(feature_slices, features);                               // Merge into single matrix
 * features.reshape(1).copyTo(features);                              // Re-interleave all channel slices
 *
 * @return
 */
void FeatureHOG::compute(
		const cv::Mat &image, cv::Mat &features, const int cell_size) const
{
	if (cvCeil(image.rows / (float) cell_size) > 0 && cvCeil(image.cols / (float) cell_size) > 0)
	{
		const auto in_pad = 1; // internal padding to prevent read/write outside matrix, though retain correct interpolation
		const auto pad_x = in_pad + ci_padx; // ci_padx x-padding for possible occlusion handling
		const auto pad_y = in_pad + ci_pady;
		const auto inv_cell_size = 1.0f / (float) cell_size;

		const int size_height = image.rows + (image.rows % cell_size != 0 ? cell_size : 0);
		const int size_width = image.cols + (image.cols % cell_size != 0 ? cell_size : 0);

		// memory for HOG features dimensions
		const int dims[] = { ci_bins * 2, size_height / cell_size + 2 * pad_y, size_width / cell_size + 2 * pad_x };

		// gradient histogram
		cv::Mat g_hist = cv::Mat(dims[1], dims[2] * dims[0], cv::DataType<float>::type, cv::Scalar::all(0));
		const size_t g_hist_s = g_hist.step1();
		float* const g_hist_p = g_hist.ptr<float>(0);

		/*
		 * Start at x=1,y=1 because we look around the pixel under consideration
		 */
		int y, x;
#ifndef DEBUG
#ifdef _OPENMP
#pragma omp parallel for private(y, x)
#endif
#endif
		for (y = 0; y < image.rows; ++y)
		{
			const auto ypp = image.ptr(std::max<int>(y - 1, 0));
			const auto ypc = image.ptr(y);
			const auto ypn = image.ptr(std::min<int>(y + 1, image.rows - 1));

			for (x = 0; x < image.cols; ++x)
			{
				// add to 4 histograms around pixel using linear interpolation
				const float xp = ((float) x + (float) 0.5) * inv_cell_size + pad_x - (float) 0.5;
				const float yp = ((float) y + (float) 0.5) * inv_cell_size + pad_y - (float) 0.5;
				const auto ixp = static_cast<int>(xp);
				const auto iyp = static_cast<int>(yp);

				PixelData pixel_data;
				pixelData(ypp, ypc, ypn, x, image, pixel_data);

				const auto adx = pixel_data.dx + 255;
				const auto ady = pixel_data.dy + 255;
				const auto bin0 = u_qtz_bins[ady][adx][0];
				const auto bin1 = u_qtz_bins[ady][adx][1];
				const float mag0 = d_qtz_mags[ady][adx][0];
				const float mag1 = d_qtz_mags[ady][adx][1];
				const float vx0 = xp - ixp;
				const float vy0 = yp - iyp;
				const float vx1 = (float) 1.0 - vx0;
				const float vy1 = (float) 1.0 - vy0;

				float* dst = g_hist_p + iyp * g_hist_s + ixp * dims[0] + bin0;
				*dst += vx1 * vy1 * mag0;
				dst = g_hist_p + iyp * g_hist_s + ixp * dims[0] + bin1;
				*dst += vx1 * vy1 * mag1;

				dst = g_hist_p + iyp * g_hist_s + (ixp + 1) * dims[0] + bin0;
				*dst += vx0 * vy1 * mag0;
				dst = g_hist_p + iyp * g_hist_s + (ixp + 1) * dims[0] + bin1;
				*dst += vx0 * vy1 * mag1;

				dst = g_hist_p + (iyp + 1) * g_hist_s + ixp * dims[0] + bin0;
				*dst += vx1 * vy0 * mag0;
				dst = g_hist_p + (iyp + 1) * g_hist_s + ixp * dims[0] + bin1;
				*dst += vx1 * vy0 * mag1;

				dst = g_hist_p + (iyp + 1) * g_hist_s + (ixp + 1) * dims[0] + bin0;
				*dst += vx0 * vy0 * mag0;
				dst = g_hist_p + (iyp + 1) * g_hist_s + (ixp + 1) * dims[0] + bin1;
				*dst += vx0 * vy0 * mag1;
			}
		}

		// normalization matrix
		cv::Mat norm(dims[1], dims[2], cv::DataType<float>::type, cv::Scalar::all(0));
		const size_t norm_s = norm.step1();
		float* const norm_p = norm.ptr<float>(0);

		// compute energy in each block by summing over all orientations
		for (y = 0; y < dims[1]; ++y)
		{
			const float* src = g_hist_p + y * g_hist_s;
			float* dst = norm_p + y * norm_s;
			float const * const dst_end = dst + dims[2];

			while (dst < dst_end)
			{
				for (int o = 0; o < ci_bins; ++o)
				{
					float value = *src + *(src + ci_bins);
					*dst += value * value;
					src++;
				}

				src += ci_bins;
				dst++;
			}
		}

		// feature matrix
		cv::Mat tmp_features(dims[1], dims[2] * ci_depth, cv::DataType<float>::type);
		const size_t tmp_feature_s = tmp_features.step1();
		float* const tmp_features_p = tmp_features.ptr<float>(0);

		features = cv::Mat(dims[1] - 2 * in_pad, (dims[2] - 2 * in_pad) * ci_depth, cv::DataType<float>::type);
		const size_t feature_s = features.step1();
		float* const features_p = features.ptr<float>(0);

#ifndef DEBUG
#ifdef _OPENMP
#pragma omp parallel for private(y, x)
#endif
#endif
		for (y = 0; y < dims[1]; ++y)
		{
			for (x = 0; x < dims[2]; ++x)
			{
				float n[4];
				int nx, ny, px, py;
				for (ny = 0; ny < 2; ++ny)
				{
					for (nx = 0; nx < 2; ++nx)
					{
						float p[4];
						for (py = -1; py < 1; ++py)
							for (px = -1; px < 1; ++px)
							{
								int ipx = std::min<int>(std::max<int>(x + nx + px, 0), dims[2] - 1);
								int ipy = std::min<int>(std::max<int>(y + ny + py, 0), dims[1] - 1);
								float val = *(norm_p + ipy * norm_s + ipx);
								p[(py + 1) * 2 + (px + 1)] = val;
							}

						float val = (float) 1.0 / std::sqrt(std::numeric_limits<float>::epsilon() + p[0] + p[1] + p[2] + p[3]);
						n[ny * 2 + nx] = val;
					}
				}

				const float* src = g_hist_p + y * g_hist_s + x * dims[0];
				float* tmp_dst = tmp_features_p + y * tmp_feature_s + x * ci_depth;
				float* dst = features_p + (y - in_pad) * feature_s + (x - in_pad) * ci_depth;

				/*
				 * contrast-sensitive features (opposing bins have not same magnitude)
				 */
				const float* src_s = src;
				float t[4];
				t[0] = 0, t[1] = 0, t[2] = 0, t[3] = 0;
				int o, i;
				for (o = 0; o < 2 * ci_bins; o++)
				{
					float h[4];
					for (i = 0; i < (int) 4; ++i)
						h[i] = std::min<float>(*src_s * n[i], ct_trunc);

					if (cb_texture_features)
					{
						t[0] += h[0];
						t[1] += h[1];
						t[2] += h[3];
						t[3] += h[2];
					}

					*tmp_dst = (float) 0.5 * (h[0] + h[1] + h[2] + h[3]) - ct_step;
					if (y >= in_pad && x >= in_pad && y < dims[1] - in_pad && x < dims[2] - in_pad)
					{
						*dst = *tmp_dst;
						++dst;
					}
					++tmp_dst;
					++src_s;
				}

				/*
				 * contrast-insensitive features (opposing bins have same magnitude)
				 */
				src_s = src;
				for (o = 0; o < ci_bins; o++)
				{
					float sum = *src_s + *(src_s + ci_bins);

					float h[4];
					for (i = 0; i < 4; ++i)
						h[i] = std::min<float>(sum * n[i], ct_trunc);

					*tmp_dst = (float) 0.5 * (h[0] + h[1] + h[2] + h[3]) - ct_step;
					if (y >= in_pad && x >= in_pad && y < dims[1] - in_pad && x < dims[2] - in_pad)
					{
						*dst = *tmp_dst;
						++dst;
					}
					++tmp_dst;
					++src_s;
				}

				/*
				 * Texture features (general pixel intensity)
				 */
				if (cb_texture_features)
				{
					for (i = 0; i < 4; ++i)
					{
						*tmp_dst = ct_texture * t[i] - ct_step;
						if (y >= in_pad && x >= in_pad && y < dims[1] - in_pad && x < dims[2] - in_pad)
						{
							*dst = *tmp_dst;
							++dst;
						}
						++tmp_dst;
					}
				}
			}
		}
	}
}

/**
 * Calculate pixel gradients dx,dy and intensity/magnitude v at point p for the given image
 *
 * @return
 */
void FeatureHOG::pixelData(
		const uchar* slp, const uchar* slc, const uchar* sln, const int x, const cv::Mat &image, PixelData &d) const
{
	const auto chn = image.channels();
	const int c = chn > 1 ? 1 : 0; // sample from Green channel only if color
	d.dx = static_cast<int>(slc[std::min<int>(x + 1, image.cols - 1) * chn + c]) - static_cast<int>(slc[std::max<int>(x - 1, 0) * chn + c]);
	d.dy = static_cast<int>(sln[x * chn + c]) - static_cast<int>(slp[x * chn + c]);
}

/**
 * Generate normalized visualization of a given HOG features matrix with given pixels per cell (cell_size)
 *
 * Example:
 * FeatureHOG descriptor;
 * cv::Mat features;
 * descriptor.compute(image, features);
 * cv::Mat pos_vis = descriptor.visualize(features, 48, 2)[0];
 * cv::imshow("HOG", pos_vis);
 * cv::waitKey();
 *
 * @return
 */
const std::vector<cv::Mat> FeatureHOG::visualise(
		const cv::Mat &features, const int cell_size, const int thickness)
{
	cv::Mat ftrs = features;
	if (cv::DataType<float>::type != features.type())
		features.convertTo(ftrs, cv::DataType<float>::type);

	cv::Mat w_pos = ftrs;
	cv::Mat w_neg = cv::Mat(w_pos.rows, w_pos.cols, w_pos.type());

	float* fsrc = w_pos.ptr<float>(0);
	float* fdst = w_neg.ptr<float>(0);

	for (size_t i = 0; i < w_pos.total(); ++i)
		*(fdst++) = -*(fsrc++);

	w_pos = fold(w_pos);
	w_neg = fold(w_neg);

	double pos_min_val, neg_min_val;
	double pos_max_val, neg_max_val;
	cv::minMaxLoc(w_pos, &pos_min_val, &pos_max_val);
	cv::minMaxLoc(w_neg, &neg_min_val, &neg_max_val);

	float scale = std::max(pos_max_val, neg_max_val);
	float factor = 255 / scale;

	const cv::Mat pos = createVis(w_pos, cell_size, thickness);
	cv::Mat pos_im;
	pos.convertTo(pos_im, CV_8UC3, factor);

	cv::Mat neg_im;
	if (neg_max_val > 0)
	{
		cv::Mat neg = createVis(w_neg, cell_size, thickness);
		neg.convertTo(neg_im, CV_8UC3, factor);
	}

	std::vector<cv::Mat> visuals(1, pos_im);
	if (!neg_im.empty())
		visuals.push_back(neg_im);

	return visuals;
}

/**
 * generate actual visualization using rotated bin images
 *
 * @return
 */
const cv::Mat FeatureHOG::createVis(
		const cv::Mat &features, const int cell_size, const int thickness)
{
	cv::Mat image;
	std::vector<cv::Mat> bins;
	cv::Mat bin = cv::Mat::zeros(cell_size, cell_size, CV_32FC3);

	cv::line(bin, cv::Point(cell_size / 2, 0), cv::Point(cell_size / 2, cell_size), cv::Scalar::all(1), thickness, CV_AA);

	double angle = 180 / (double) ci_bins;

	bins.push_back(bin);
	int i;
	for (i = 1; i < ci_bins; ++i)
		bins.push_back(rotateImage(bin, -i * angle));

	image = cv::Mat::zeros(cell_size * features.rows, cell_size * (features.cols / ci_bins), CV_32FC3);

	const unsigned int feature_s = features.step1();
	const float* fsrc = features.ptr<float>(0);

	int x, y;
	for (y = 0; y < features.rows; y++)
	{
		int ys = y * cell_size;

		for (x = 0; x < features.cols / ci_bins; x++)
		{
			const float* src = fsrc + y * feature_s + x * ci_bins;

			int xs = x * cell_size;
			cv::Mat block = image(cv::Rect(xs, ys, cell_size, cell_size));

			for (int bin = 0; bin < ci_bins; bin++)
				block = block + (bins.at(bin) * std::max<float>(*(src++), (float) 0));
		}
	}

	return image;
}

/**
 * Fold the constrast sensitive and constrast insensitive features together
 *
 * @return
 */
const cv::Mat FeatureHOG::fold(
		const cv::Mat &features)
{
	cv::Mat f;
	const int width = features.cols / ci_depth;
	f = cv::Mat::zeros(features.rows, width * ci_bins, features.type());

	const float* fsrc = features.ptr<float>(0);
	float* fdst = f.ptr<float>(0);
	const unsigned int src_s = features.step1();
	const unsigned int dst_s = f.step1();

	int x, y;
	for (x = 0; x < width; ++x)
	{
		for (y = 0; y < features.rows; ++y)
		{
			const float* src = fsrc + y * src_s + x * ci_depth;
			float* dst = fdst + y * dst_s + x * ci_bins;

			float* dst_s = dst;
			for (int z = 0; z < ci_bins; ++z)
				*(dst_s++) += std::max<float>(*(src++), (float) 0);

			dst_s = dst;
			for (int z = ci_bins; z < ci_bins * 2; ++z)
				*(dst_s++) += std::max<float>(*(src++), (float) 0);

			dst_s = dst;
			for (int z = ci_bins * 2; z < ci_bins * 3; ++z)
				*(dst_s++) += std::max<float>(*(src++), (float) 0);
		}
	}

	return f;
}

/**
 * Rotate an image with a line a certain angle
 *
 * @return
 */
cv::Mat FeatureHOG::rotateImage(
		const cv::Mat& source, double angle)
{
	cv::Point2f src_center(source.cols / 2.f, source.rows / 2.f);
	cv::Mat rot_mat = getRotationMatrix2D(src_center, angle, 1.0);
	cv::Mat dst;
	warpAffine(source, dst, rot_mat, source.size());
	return dst;
}

// end ************************************ Felzenszwalb HOG ******************************************* end

} /* namespace nl_uu_science_gmt */
