/*
 * FeatureHOG.h
 *
 *  Created on: Feb 15, 2013
 *      Author: coert
 */

#ifndef FEATUREHOG_H_
#define FEATUREHOG_H_

#include <opencv2/core/core.hpp>
#include <vector>

namespace nl_uu_science_gmt
{

class FeatureHOG
{
private:
	uchar* u_qtz_bins;
	float* d_qtz_mags;

	const int ci_padx, ci_pady, ci_bins, ci_lbp_bins;
	const float ct_trunc, ct_texture, ct_step;
	const bool cb_texture_features;
	const int ci_depth;

	struct PixelData
	{
		int dx;
		int dy;
	};

	inline const cv::Mat fold(
			const cv::Mat &);
	inline cv::Mat rotateImage(
			const cv::Mat &, double);
	const cv::Mat createVis(
			const cv::Mat &, const int, const int);
	inline void pixelData(
			const uchar*, const uchar*, const uchar*, const int, const cv::Mat &, PixelData &) const;

public:
	FeatureHOG();
	virtual ~FeatureHOG();

	void compute(
			const cv::Mat &, cv::Mat &, const int) const;
	const std::vector<cv::Mat> visualise(
			const cv::Mat &, const int, const int = 2);

	const int getDepth() const
	{
		return ci_depth;
	}

	const bool useTextureFeatures() const
	{
		return cb_texture_features;
	}
};

}
/* namespace nl_uu_science_gmt */
#endif /* FEATUREHOG_H_ */
