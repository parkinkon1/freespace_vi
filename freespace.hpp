#pragma once

#include "opencv/cv.h"
#include "opencv2/opencv.hpp"
using namespace std;
using namespace cv;

class FreeSpace
{
public:
	FreeSpace();
	~FreeSpace();

	struct CameraParam {
		float fu, fv;
		float cu, cv;
		float base;
	};

	void initialize(float base, int imageWidth, int imageHeight);
	cv::Mat detect(const cv::Mat& in_disp);
	cv::Mat draw(const cv::Mat& imgray, const cv::Mat& imdisp, const cv::Mat& imfreespace);

private:

	void preprocess(const cv::Mat& in, cv::Mat& out);
	void computeIDH(const cv::Mat& in, cv::Mat& out_mask, uint16_t* out_idh);
	void removeObstacle(const uint16_t* in_idh, cv::Mat& in_mask);
	void estimateLRSmodel(const cv::Mat& in_mask, int* out_vdisp);
	void estimteFreeSpace(
		const cv::Mat in, const uint16_t* in_idh,
		const int* in_vdisp, cv::Mat& out_mask);

	// User-defined parameters
	int _MAX_DISP, _BOUND_LEFT, _BOUND_RIGHT, _BOUND_TOP, _BOUND_BOTTOM;
	int _OBS_REMOV_MIN_OBJ_HEIGHT, _OBS_REMOV_HEIGHT_OFFSET;
	float _OBS_REMOV_OBJ_HEIGHT, _OBS_REMOV_DETECT_PROB;
	float _VDISP_DP_WEIGHT_DATA, _VDISP_DP_WEIGHT_SMOOTH,
		_VDISP_DP_WEIGHT_GROUND, _VDISP_DP_PARAM_SMOOTH,
		_VDISP_DP_PARAM_GROUND;
	float _FSE_DP_COST_SMOOTH_MAX, _FSE_DP_PARAM_COST3;

	// Camera parameters
	int _imageWidth, _imageHeight;
	cv::Size _imageSize;
	CameraParam _param;

	// Images
	cv::Mat _in, _obs_disp;

	// IDH computation
	uint16_t* _idh;

	// Obstacle removal
	int _numDisp, _obs_min_obs_height;
	int* _obs_remov_numPixelObjFunc;
	int* _obs_remov_thr_func;

	// V-Disparity computation
	int* _vdisp_weight_map;
	float* _vdisp_dp_smooth_table;
	float* _vdisp_dp_ground_table;
	int* _vdisp;
	int* _vdispf;
	int* _vdispm;
	float* _vdisp_dp_datacost;
	float* _vidsp_dp_cost;
	int* _vdisp_dp_idx;
	int* _vdisp_min_row_disp;
	int* _vdisp_max_row_disp;
	int _vdisp_horizon;

	// Free space estimation
	int *_fse_v1, *_fse_v2, *_fse_v21;
	float* _fse_data_cost;
	float* _fse_smooth_cost_table;
	float* _fse_data_cost3_table;
	int* _fse_tracker;
	float* _fse_costmap;
	int* _fse_trajectory;
};