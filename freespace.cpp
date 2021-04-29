#include "freespace.hpp"
#include <algorithm>

// #define FSE_USE_WEIGHTMAP_VDCOMPUT // V-Disparity weight map for robust estimation

FreeSpace::FreeSpace() 
{
	_idh = NULL;
	_obs_remov_numPixelObjFunc = NULL;
	_obs_remov_thr_func = NULL;
	_vdispm = NULL;
	_vdispf = NULL;
	_vdisp = NULL;
	_vdisp_weight_map = NULL;
	_vdisp_dp_smooth_table = NULL;
	_vdisp_dp_ground_table = NULL;
	_vdisp_dp_datacost = NULL;
	_vidsp_dp_cost = NULL;
	_vdisp_dp_idx = NULL;
	_vdisp_min_row_disp = NULL;
	_vdisp_max_row_disp = NULL;
	_fse_data_cost = NULL;
	_fse_smooth_cost_table = NULL;
	_fse_data_cost3_table = NULL;
	_fse_v1 = NULL;
	_fse_v2 = NULL;
	_fse_v21 = NULL;
	_fse_tracker = NULL;
	_fse_costmap = NULL;
	_fse_trajectory = NULL;
}

FreeSpace::~FreeSpace() 
{
	if (_idh)
		delete[] _idh;
	if (_obs_remov_numPixelObjFunc)
		delete[] _obs_remov_numPixelObjFunc;
	if (_obs_remov_thr_func)
		delete[] _obs_remov_thr_func;
	if (_vdispm)
		delete[] _vdispm;
	if (_vdispf)
		delete[] _vdispf;
	if (_vdisp)
		delete[] _vdisp;
	if (_vdisp_weight_map)
		delete[] _vdisp_weight_map;
	if (_vdisp_dp_smooth_table)
		delete[] _vdisp_dp_smooth_table;
	if (_vdisp_dp_ground_table)
		delete[] _vdisp_dp_ground_table;
	if (_vdisp_dp_datacost)
		delete[] _vdisp_dp_datacost;
	if (_vidsp_dp_cost)
		delete[] _vidsp_dp_cost;
	if (_vdisp_dp_idx)
		delete[] _vdisp_dp_idx;
	if (_vdisp_min_row_disp)
		delete[] _vdisp_min_row_disp;
	if (_vdisp_max_row_disp)
		delete[] _vdisp_max_row_disp;
	if (_fse_data_cost)
		delete[] _fse_data_cost;
	if(_fse_smooth_cost_table)
		delete[] _fse_smooth_cost_table;
	if (_fse_data_cost3_table)
		delete[] _fse_data_cost3_table;
	if (_fse_v1)
		delete[] _fse_v1;
	if (_fse_v2)
		delete[] _fse_v2;
	if (_fse_v21)
		delete[] _fse_v21;
	if (_fse_tracker)
		delete[] _fse_tracker;
	if (_fse_costmap)
		delete[] _fse_costmap;
	if (_fse_trajectory)
		delete[] _fse_trajectory;
}

void FreeSpace::initialize(float base, int imageWidth, int imageHeight)
{
	_param.cu = (float)(imageWidth/2);
	_param.cv = (float)(imageHeight/2);
	_param.base = base;
	_imageWidth = imageWidth;
	_imageHeight = imageHeight;
	_imageSize = cv::Size(imageWidth, imageHeight);

	// User-defined parameters - Data1
	_MAX_DISP = 47;
	_BOUND_LEFT = 60;
	_BOUND_RIGHT = 30;
	_BOUND_TOP = 30;
	_BOUND_BOTTOM = 30;

	_OBS_REMOV_MIN_OBJ_HEIGHT = 30;
	_OBS_REMOV_OBJ_HEIGHT = 0.5f;
	_OBS_REMOV_DETECT_PROB = 0.7f;
	_OBS_REMOV_HEIGHT_OFFSET = 150;

	_VDISP_DP_WEIGHT_DATA = 1.f;
	_VDISP_DP_WEIGHT_SMOOTH = 0.3f;
	_VDISP_DP_WEIGHT_GROUND = 0.2f;
	_VDISP_DP_PARAM_SMOOTH = 5.f;
	_VDISP_DP_PARAM_GROUND = 15.f;

	_FSE_DP_COST_SMOOTH_MAX = 8.f;
	_FSE_DP_PARAM_COST3 = 5.f;

	/*
	// User-defined parameters - Data2
	_MAX_DISP = 85;
	_BOUND_LEFT = 10;
	_BOUND_RIGHT = 10;
	_BOUND_TOP = 10;
	_BOUND_BOTTOM = 10;

	_OBS_REMOV_MIN_OBJ_HEIGHT = 10;// 1;
	_OBS_REMOV_OBJ_HEIGHT = 0.01f;
	_OBS_REMOV_DETECT_PROB = 0.7f;
	_OBS_REMOV_HEIGHT_OFFSET = 50;

	_VDISP_DP_WEIGHT_DATA = 1.f;
	_VDISP_DP_WEIGHT_SMOOTH = 0.15f;// 0.3f;
	_VDISP_DP_WEIGHT_GROUND = 0.2f;
	_VDISP_DP_PARAM_SMOOTH = 2.5f;// 5.f;
	_VDISP_DP_PARAM_GROUND = 15.f;

	_FSE_DP_COST_SMOOTH_MAX = 5.0f;// 8.f;
	_FSE_DP_PARAM_COST3 = 5.f;
	*/

	// Preprocess
	_in = cv::Mat(imageHeight, imageWidth, CV_32FC1);

	// IDH computation
	_numDisp = _MAX_DISP + 1;
	_idh = new uint16_t[imageHeight*imageWidth*_numDisp];

	// Obstacle removal
	_obs_disp = cv::Mat(imageHeight, imageWidth, CV_8UC1);
	_obs_disp.setTo(0);
	_obs_remov_numPixelObjFunc = new int[_numDisp];
	for (int i = 0; i < _numDisp; i++)
	{
		_obs_remov_numPixelObjFunc[i] = (int)roundf((float)i * _OBS_REMOV_OBJ_HEIGHT / _param.base);

		if (_obs_remov_numPixelObjFunc[i] < _OBS_REMOV_MIN_OBJ_HEIGHT)
			_obs_remov_numPixelObjFunc[i] = _OBS_REMOV_MIN_OBJ_HEIGHT;
	}

	_obs_remov_thr_func = new int[_numDisp];
	for (int i = 0; i < _numDisp; i++)
	{
		_obs_remov_thr_func[i] = std::min(
			(int)roundf((float)_obs_remov_numPixelObjFunc[i] * 
				_OBS_REMOV_DETECT_PROB), 60);
	}

	_obs_min_obs_height = (int)ceilf(_param.cv - _OBS_REMOV_HEIGHT_OFFSET);

	// V-Disparity computation
	_vdispm = new int[_imageHeight*_numDisp];
	_vdispf = new int[_imageHeight*_numDisp];
	_vdisp = new int[_imageHeight];

#ifdef FSE_USE_WEIGHTMAP_VDCOMPUT
	_vdisp_weight_map = new int[imageHeight*imageWidth];
	int* ptri = _vdisp_weight_map;
	for (int i = 0; i < imageWidth*imageHeight; i++, ptri++)
		*ptri = 1;

	int px = (int)roundf(_param.cu);
	int py = (int)roundf(_param.cv);
	int x2 = (int)roundf(_param.cu - (float)imageWidth*0.25f);
	int x3 = (int)roundf(_param.cu + (float)imageWidth*0.25f);
	int y2 = imageHeight;
	int y3 = imageHeight;
	float a2 = (float)(py - y2) / (float)(px - x2);
	float a3 = (float)(py - y3) / (float)(px - x3);
	float b2 = (float)y2 - a2*(float)x2;
	float b3 = (float)y3 - a3*(float)x3;
	for (int i = x2; i <= px; i++)
	{
		for (int j = (int)roundf(a2*(float)i + b2); j < imageHeight; j++)
			_vdisp_weight_map[j*imageWidth + i] = 2;
	}
	for (int i = px; i <= x3; i++)
	{
		for (int j = (int)roundf(a3*(float)i + b3); j < imageHeight; j++)
			_vdisp_weight_map[j*imageWidth + i] = 2;
	}
#endif
	
	_vdisp_dp_smooth_table = new float[_numDisp];
	_vdisp_dp_ground_table = new float[_numDisp];
	for (int i = 0; i < _numDisp; i++)
	{
		_vdisp_dp_smooth_table[i] = _VDISP_DP_WEIGHT_SMOOTH *
			(1.f - expf(-((float)i) / _VDISP_DP_PARAM_SMOOTH));
		_vdisp_dp_ground_table[i] = _VDISP_DP_WEIGHT_GROUND *
			(1.f - expf(-((float)i) / _VDISP_DP_PARAM_GROUND));
	}
	_vidsp_dp_cost = new float[_numDisp*_imageHeight];
	_vdisp_dp_datacost = new float[_numDisp*_imageHeight];
	_vdisp_dp_idx = new int[_numDisp*_imageHeight];
	_vdisp_min_row_disp = new int[_imageHeight];
	_vdisp_max_row_disp = new int[imageHeight];

	// Free space estimation
	_fse_smooth_cost_table = new float[_numDisp];
	_fse_data_cost3_table = new float[_numDisp];
	_fse_data_cost = new float[_numDisp*_imageWidth];
	float r = _FSE_DP_COST_SMOOTH_MAX / 10.f;
	for (int i = 0; i < _numDisp; i++)
	{
		if (i > 10)
		{
			_fse_smooth_cost_table[i] = _FSE_DP_COST_SMOOTH_MAX;
		}
		else
		{
			_fse_smooth_cost_table[i] = (float)i *r;
		}
		_fse_data_cost3_table[i] = expf(-(float)i / _FSE_DP_PARAM_COST3);
	}
	_fse_v1 = new int[_numDisp];
	_fse_v2 = new int[_numDisp];
	_fse_v21 = new int[_numDisp];
	_fse_tracker = new int[_numDisp*_imageWidth];
	_fse_costmap = new float[_numDisp*_imageWidth];
	_fse_trajectory = new int[_imageWidth];
}

cv::Mat FreeSpace::detect(const cv::Mat& in_disp)
{
	cv::Mat out_mask(_imageHeight, _imageWidth, CV_8UC1);
	out_mask.setTo(0);

	// Preprocessing
	preprocess(in_disp, _in);

	// Compute IDH
	computeIDH(_in, _obs_disp, _idh);

	// Remove obstacle regions
	removeObstacle(_idh, _obs_disp);

	// Estimate a longitudinal road surface model
	estimateLRSmodel(_obs_disp, _vdisp);

	// Estimate free space
	estimteFreeSpace(_in, _idh, _vdisp, out_mask);

	return out_mask;
}

void FreeSpace::preprocess(const cv::Mat& in, cv::Mat& out)
{
	// Set out-of-range disparity values to -1
	const float* ptrI = in.ptr<float>(0);
	float* ptrO = out.ptr<float>(0);
	for (int i = 0; i < _imageHeight*_imageWidth; i++, ptrI++, ptrO++)
	{
		if (*ptrI > _MAX_DISP)
			*ptrO = -1.f;
		else
		{
			if (*ptrI <= 0)
				*ptrO = -1.f;
			else
				*ptrO = *ptrI;
		}
	}
}

void FreeSpace::computeIDH(const cv::Mat& in, cv::Mat& out_mask, uint16_t* out_idh)
{
	memset(out_idh, 0, sizeof(uint16_t)*_imageHeight*_imageWidth*_numDisp);
	int wn = _imageWidth*_numDisp;
	
	int x = _BOUND_LEFT;
	int y = _BOUND_TOP;
	uint32_t idx = y*_imageWidth + x;
	int xe = _imageWidth - _BOUND_RIGHT;
	const float* ptrI = &in.at<float>(idx);
	uchar* ptrO = &out_mask.at<uchar>(idx);
	uint16_t* ptrD = out_idh + y*wn + x*_numDisp;	

	// First row
	for (; x < xe; x++, ptrI++, ptrO++, ptrD += _numDisp)
	{
		int i = (int)roundf(*ptrI);

		if (*ptrI <= _MAX_DISP)
		{
			if (*ptrI > 0)
			{
				*(ptrD + i) = 1;
				*ptrO = i;
			}
			else
			{
				*ptrO = 255;
			}
		}
		else
		{
			*ptrO = 255;
		}
	}

	// From second to bottom
	for (y = _BOUND_TOP + 1; y < _imageHeight - _BOUND_BOTTOM; y++)
	{
		uint32_t _i1 = y*wn;
		uint32_t _i2 = _i1 - wn;
		memcpy(out_idh + _i1, out_idh + _i2, sizeof(uint16_t)*wn);

		x = _BOUND_LEFT;
		idx = y*_imageWidth + x;
		const float* ptrI = &in.at<float>(idx);
		uchar* ptrO = &out_mask.at<uchar>(idx);
		uint16_t* ptrD = out_idh + y*wn + x*_numDisp;

		for (; x < xe; x++, ptrI++, ptrO++, ptrD += _numDisp)
		{
			int i = (int)roundf(*ptrI);

			if (*ptrI <= _MAX_DISP)
			{
				if (*ptrI > 0)
				{
					*(ptrD + i) += 1;
					*ptrO = i;
				}
				else
				{
					*ptrO = 255;
				}
			}
			else
			{
				*ptrO = 255;
			}
		}
	}
}

void FreeSpace::removeObstacle(const uint16_t* in_idh, cv::Mat& in_mask)
{
	int wn = _imageWidth*_numDisp;

	for (int x = _BOUND_LEFT; x < _imageWidth - _BOUND_RIGHT; x++)
	{
		int y = _imageHeight - _BOUND_BOTTOM - 1;
		int idx = y*_imageWidth + x;
		bool bCheck = false;

		for (; y >= _obs_min_obs_height; y--, idx -= _imageWidth)
		{
			int i = (int)roundf(in_mask.at<uchar>(idx));

			if (i < _numDisp)
			{
				if (i >= 0)
				{
					int j = 0;
					int l = std::max(y - (int)_obs_remov_numPixelObjFunc[i], 1);
					for (int k = 0; k <= 1; k++)
					{
						if (i + k < _numDisp)
						{
							int o = x*_numDisp + i + k;
							j += (_idh[y*wn + o] -
								_idh[l*wn + o]);
						}							
					}
					if (j >= _obs_remov_thr_func[i])
						bCheck = true;
				}
			}
			if (bCheck)
			{
				uchar* ptr = &in_mask.at<uchar>(x);
				for (i = 0; i <= y; i++, ptr += _imageWidth)
					*ptr = 255;
				break;
			}
		}
		if (!bCheck)
		{
			uchar* ptr = &in_mask.at<uchar>(x);
			for (int i = 0; i < _obs_min_obs_height; i++, ptr += _imageWidth)
				*ptr = 255;
		}
	}
}

void FreeSpace::estimateLRSmodel(const cv::Mat& in_mask, int* out_vdisp)
{
	// V-Disparity map computation
	memset(_vdispm, 0, sizeof(int)*_numDisp*_imageHeight);
	for (int x = _BOUND_LEFT; x < _imageWidth - _BOUND_RIGHT; x++)
	{
		for (int y = _imageHeight - _BOUND_BOTTOM - 1; y >= _obs_min_obs_height; y--)
		{
			int i = (int)in_mask.at<uchar>(y*_imageWidth + x);

			if (i < _numDisp)
			{
				if (i >= 0)
				{
#ifdef FSE_USE_WEIGHTMAP_VDCOMPUT
					_vdispm[y*_numDisp + i] += _vdisp_weight_map[y*_imageWidth + x];
#else
					_vdispm[y*_numDisp + i]++;
#endif
				}
			}
		}
	}

	// Compute data costs and row-wise minimum disparity values
	for (int v = _BOUND_TOP; v < _imageHeight - _BOUND_BOTTOM; v++)
	{
		int maxD = -1;
		int l = v*_numDisp;
		for (int d = 0; d < _numDisp; d++)
			if (maxD < _vdispm[l + d])
				maxD = _vdispm[l + d];
		if (maxD < 30)
			maxD = 30;
		_vdisp_max_row_disp[v] = maxD;
	}

	for (int v = _BOUND_TOP; v < _imageHeight - _BOUND_BOTTOM; v++)
		_vdisp_min_row_disp[v] = -1;

	for (int v = _BOUND_TOP; v <= _imageHeight - _BOUND_BOTTOM - 1; v++)
	{
		int l = v*_numDisp;
		for (int d = 0; d < _numDisp; d++)
		{
			// Data cost computation
			_vdisp_dp_datacost[l + d] = _VDISP_DP_WEIGHT_DATA * (
				1.f - (float)_vdispm[l + d] / (float)_vdisp_max_row_disp[v]);

			// Minimum disparity estimation
			if (_vdisp_min_row_disp[v] == -1)
				if (_vdispm[l + d] > 0)
					_vdisp_min_row_disp[v] = d;
		}
	}

	// Dynamic programming
	memset(_vidsp_dp_cost, 0, sizeof(float)*_imageHeight*_numDisp);
	int* ptr = _vdisp_dp_idx;
	for (int i = 0; i < _imageHeight*_numDisp; i++)
		*(ptr++) = -1;

	int v1 = _imageHeight - _BOUND_BOTTOM - 1;
	int l1 = v1*_numDisp;
	for (int d = 0; d < _numDisp; d++)
	{
		if (d <= _vdisp_min_row_disp[v1])
			_vidsp_dp_cost[l1 + d] = _vdisp_dp_datacost[l1 + d] + _VDISP_DP_WEIGHT_GROUND;
		else
			_vidsp_dp_cost[l1 + d] = _vdisp_dp_datacost[l1 + d] + _vdisp_dp_ground_table[d - _vdisp_min_row_disp[v1]];
	}

	for (int v = _imageHeight - _BOUND_BOTTOM - 2;
		v >= _BOUND_TOP; v--)
	{
		int l = v*_numDisp;
		int k = (v + 1)*_numDisp;

		for (int d = 0; d < _numDisp; d++)
		{
			float tmpMinCost = 10000000.f;
			int tmpMinIdx = -1;

			if (_vdisp_min_row_disp[v] == -1)
			{
				for (int dp = d; dp < _numDisp; dp++)
				{
					float tmpCost = _vidsp_dp_cost[k + dp] + _vdisp_dp_datacost[l + d] +
						_VDISP_DP_WEIGHT_GROUND + _vdisp_dp_smooth_table[dp - d];

					if (tmpCost < tmpMinCost)
					{
						tmpMinIdx = dp;
						tmpMinCost = tmpCost;
					}
				}
			}
			else
			{
				float tmpGroundCost = _vdisp_dp_ground_table[abs(d - _vdisp_min_row_disp[v])];
				for (int dp = d; dp < _numDisp; dp++)
				{
					float tmpCost = _vidsp_dp_cost[k + dp] + _vdisp_dp_datacost[l + d] +
						tmpGroundCost + _vdisp_dp_smooth_table[dp - d];
					if (tmpCost < tmpMinCost)
					{
						tmpMinIdx = dp;
						tmpMinCost = tmpCost;
					}
				}
			}

			_vidsp_dp_cost[l + d] = tmpMinCost;
			_vdisp_dp_idx[l + d] = tmpMinIdx;
		}
	}

	// V-Disparity function estimation
	for (int i = 0; i < _imageHeight; i++)
		_vdispf[i] = -1;

	v1 = _BOUND_TOP;
	l1 = v1*_numDisp;
	float tmpMinCost = 1000000000.f;
	int d1 = -1;
	for (int dp = 0; dp < _numDisp; dp++)
	{
		if (_vidsp_dp_cost[l1 + dp] < tmpMinCost)
		{
			tmpMinCost = _vidsp_dp_cost[l1 + dp];
			d1 = dp;
		}
	}
	_vdispf[v1] = d1;

	for (int v = _BOUND_TOP + 1; v < _imageHeight - _BOUND_BOTTOM; v++)
		_vdispf[v] = _vdisp_dp_idx[(v - 1)*_numDisp + _vdispf[v - 1]];

	// Horizon line selection
	int vh = _obs_min_obs_height;
	for (int v = _imageHeight - _BOUND_BOTTOM - 1;
		v >= _obs_min_obs_height; v--)
	{
		int vi = std::max(v - _obs_remov_thr_func[_vdispf[v]],
			_BOUND_TOP);
		if ((_vdispf[v] - _vdispf[vi]) == 0)
		{
			vh = v;
			break;
		}

	}

	for (int v = _BOUND_TOP; v <= vh; v++)
		_vdispf[v] = -1;

	// Invert V-D function to D-V function
	for (int d = 0; d < _numDisp; d++)
		out_vdisp[d] = -2;

	for (int d = 0; d < _numDisp; d++)
	{
		for (int yi = _obs_min_obs_height; yi < _imageHeight - _BOUND_BOTTOM; yi++)
		{
			if (_vdispf[yi] == d)
			{
				out_vdisp[d] = yi;
				break;
			}
		}
	}

	int y1 = _imageHeight - _BOUND_BOTTOM;
	for (int d = _numDisp - 1; d >= 0; d--)
	{
		if (out_vdisp[d] == -2)
			out_vdisp[d] = y1;
		else
			break;
	}
	_vdisp_horizon = _obs_min_obs_height;
	for (int d = 0; d < _numDisp; d++)
	{
		if (out_vdisp[d] == -2)
			out_vdisp[d] = -1;
		else
		{
			_vdisp_horizon = d;
			break;
		}
	}

	for (int d = 1; d < _numDisp - 1; d++)
	{
		if (out_vdisp[d] == -2)
		{
			float fx = -1000.f;
			float fy = -1000.f;
			float fi, fj;

			for (int i = std::max(d - 1, 0); i >= 0; i--)
			{
				if (out_vdisp[i] >= 0)
				{
					fx = (float)out_vdisp[i];
					fi = (float)i;
					break;
				}
			}
			for (int i = std::min(d + 1, _numDisp - 1); i < _numDisp; i++)
			{
				if (out_vdisp[i] >= 0)
				{
					fy = (float)out_vdisp[i];
					fj = (float)i;
					break;
				}
			}
			if (fx > -999.f)
			{
				if (fy > -999.f)
					out_vdisp[d] = (int)roundf((fy - fx) / (fj - fi) * ((float)d - fi) + fx);
				else
					out_vdisp[d] = (int)fx;
			}
			else
				if (fy > -999.f)
					out_vdisp[d] = (int)fy;
		}
	}
}

void FreeSpace::estimteFreeSpace(const cv::Mat in, const uint16_t* in_idh,
	const int* in_vdisp, cv::Mat& out_mask)
{
	// Cost computation
	for (int d = 0; d < _numDisp; d++)
	{
		_fse_v1[d] = std::min(std::max(in_vdisp[d] - _obs_remov_thr_func[d],
			_BOUND_TOP), _imageHeight - _BOUND_BOTTOM - 1);
		_fse_v2[d] = std::min(std::max(in_vdisp[d], _BOUND_TOP), _imageHeight - _BOUND_BOTTOM - 1);
		_fse_v21[d] = _fse_v2[d] - _fse_v1[d];
	}

	float* SC = _fse_smooth_cost_table;
	float* DC3 = _fse_data_cost3_table;

	float* DC = _fse_data_cost;
	memset(DC, 0, sizeof(float)*_numDisp*_imageWidth);

	for (int x = _BOUND_LEFT; x < _imageWidth - _BOUND_RIGHT; x++)
	{
		int DC3s = 0;
		int DC3i = -1;

		for (int d = _numDisp - 1; d >= 0; d--)
		{
			int l = d*_imageWidth;
			if (in_vdisp[d] >= 0)
			{
				float DC1, DC2, DC2c;
				float fd = in.at<float>(_fse_v2[d] *_imageWidth + x);

				if (fd > 0)
					DC1 = exp(-fabsf(fd - (float)d) / 8.f);
				else
					DC1 = 0.7f;

				if (d > _vdisp_horizon)
					DC2c = (float)(in_idh[_fse_v2[d] * _imageWidth*_numDisp + x*_numDisp + d] -
						in_idh[_fse_v1[d] * _imageWidth*_numDisp + x*_numDisp + d]);
				else
				{
					DC2c = 0.f;
					for (int di = 0; di <= d; di++)
						DC2c += (float)(in_idh[_fse_v2[d] * _imageWidth*_numDisp + x*_numDisp + di] -
							in_idh[_fse_v1[d] * _imageWidth*_numDisp + x*_numDisp + di]);
				}

				if (d + 1 < _numDisp)
					DC2c += (float)(in_idh[_fse_v2[d] * _imageWidth*_numDisp + x*_numDisp + d + 1] -
						in_idh[_fse_v1[d] * _imageWidth*_numDisp + x*_numDisp + d + 1]);
				DC2 = exp(-(1 - DC2c / (float)_fse_v21[d]) / 0.5f);

				if (DC3s == 0)
				{
					if (DC2 > 0.7f)
					{
						DC3s = 1;
						DC3i++;
						DC[l + x] = 1.f - DC1*DC2*DC3[DC3i];
					}
					else
						DC[l + x] = 1 - DC1*DC2;
				}
				else
				{
					DC3i++;
					DC[l + x] = 1 - DC1*DC2*DC3[DC3i];
				}

			}
			else
				DC[l + x] = 1.f;
		}
	}

	// Cost aggregation
	for (int x = _BOUND_LEFT; x < _imageWidth - _BOUND_RIGHT; x++)
	{
		for (int d = 0; d < _numDisp; d++)
		{
			int l = d*_imageWidth + x;
			if (x == _BOUND_LEFT)
				_fse_costmap[l] = DC[l];
			else
			{
				float ft1 = 10000000000000000.f;
				int t1 = -1;
				for (int di = 0; di < _numDisp; di++)
				{
					float fi = DC[l] + SC[abs(d - di)] +
						_fse_costmap[di*_imageWidth + x - 1];
					if (fi < ft1)
					{
						ft1 = fi;
						t1 = di;
					}
				}
				_fse_costmap[l] = ft1;
				_fse_tracker[l] = t1;
			}
		}
	}

	// Minimum cost trajectory tracking
	int t1 = -1;
	float ft1 = 100000000000000000.f;
	for (int d = 0; d < _numDisp; d++)
	{
		if (_fse_costmap[(d + 1)*_imageWidth - _BOUND_RIGHT - 1] < ft1)
		{
			ft1 = _fse_costmap[(d + 1)*_imageWidth - _BOUND_RIGHT - 1];
			t1 = d;
		}
	}

	_fse_trajectory[_imageWidth - _BOUND_RIGHT - 1] = t1;
	for (int i = _imageWidth - _BOUND_RIGHT - 2; i >= _BOUND_LEFT; i--)
		_fse_trajectory[i] = _fse_tracker[_fse_trajectory[i + 1] * _imageWidth + i + 1];
	
	// Free space estimation
	int k = _imageHeight - _BOUND_BOTTOM;
	for (int x = _BOUND_LEFT; x < _imageWidth - _BOUND_RIGHT; x++)
	{
		int m = 0;
		int l = in_vdisp[_fse_trajectory[x]];
		if (l > -1)
		{
			if (l < k)
			{
				for (int y = l; y < _imageHeight - _BOUND_BOTTOM; y++)
					out_mask.at<uchar>(y, x) = 255;
				for (int y = _BOUND_TOP; y < l; y++)
					out_mask.at<uchar>(y, x) = 0;
				m = 1;
			}
		}
		if (m == 0)
			for (int y = _BOUND_TOP; y < _imageHeight - _BOUND_BOTTOM; y++)
				out_mask.at<uchar>(y, x) = 0;
	}
}

Mat FreeSpace::draw(const cv::Mat& imgray, const Mat& imdisp, const Mat& imfreespace)
{
	const float MAX_DISP = 85.f;
	int h = imgray.rows;
	int w = imgray.cols;

	Mat imrgb;
	cvtColor(imgray, imrgb, COLOR_GRAY2RGB);

	cv::Mat imdispm(h, w, CV_8UC1);
	float ratio = 255.f / MAX_DISP;
	for (int v = 0; v < h; v++)
		for (int u = 0; u < w; u++)
			imdispm.at<uchar>(v, u) = (uchar)roundf(std::min(std::max(imdisp.at<float>(v, u), 0.f), MAX_DISP)*ratio);
	Mat imdispc;
	cvtColor(imdispm, imdispc, COLOR_GRAY2RGB);
	
	Mat imfsc;
	cvtColor(imfreespace, imfsc, COLOR_GRAY2RGB);

	Mat mix;
	imrgb.copyTo(mix);
	for (int v = 0; v < h; v++)
	{
		for (int u = 0; u < w; u++)
		{
			if (imfreespace.at<uchar>(v, u) > 0)
			{
				mix.at<cv::Vec3b>(v, u)[1] = (uchar)std::min(std::max((int)mix.at<cv::Vec3b>(v, u)[1] + 100, 0), 255);
			}
		}
	}
	
	Mat v1, v2;
	hconcat(imrgb, imdispc, v1);
	hconcat(imfsc, mix, v2);

	Mat sum;
	vconcat(v1, v2, sum);

	Mat out;
	resize(sum, out, cv::Size(w / 1, h / 1));
	
	return out;
}