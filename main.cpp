// Free Space Estimation
// Before running the program, please check the parameters on Line93-136 in freespace.cpp.

#include <iostream>
#include <chrono> // To check elapsed times

// Opencv
//#include "opencv/cv.h"
//#include "opencv/highgui.h"
//#include "opencv2/opencv.hpp"
//using namespace cv;
//using namespace std;

#ifdef _DEBUG
#pragma comment(lib, "opencv_core2413d.lib")
#pragma comment(lib, "opencv_highgui2413d.lib")
#pragma comment(lib, "opencv_imgproc2413d.lib")
#pragma comment(lib, "freespace_d.lib")
#else
#pragma comment(lib, "opencv_core2413.lib")
#pragma comment(lib, "opencv_highgui2413.lib")
#pragma comment(lib, "opencv_imgproc2413.lib")
#pragma comment(lib, "freespace.lib")
#endif

#include "freespace.hpp"
//#include <cstdio>

cv::Mat loadDisp(int imageHeight, int imageWidth, char* filepath)
{
	FILE *fp = fopen(filepath, "rb");

	if (!fp)
		return cv::Mat();

	cv::Mat im(imageHeight, imageWidth, CV_32FC1);
	fread(&im.at<float>(0), 4, imageHeight*imageWidth, fp);
	fclose(fp);	

	return im;
}

int main()
{
    std::cout << "[+] start main" << std::endl;

	// << Data1 >>
	// Set paths
	char imagePath[] = "/home/q10/ws/etc/FreeSpaceEstimation_q10/data1/gray/%04d.bmp";
	char dispPath[] = "/home/q10/ws/etc/FreeSpaceEstimation_q10/data1/disp/%04d.dat";
	char resultPath[] = "/home/q10/ws/etc/FreeSpaceEstimation_q10/data1/result/%04d.jpg";
	// Number of input data
	int numData = 6;
	// Input camera parameters
	float _base = 0.15984f;
	int _imageWidth = 1280;
	int _imageHeight = 672;
	
	// Initialization
    std::cout << "[+] Free-space module initialize" << std::endl;
	FreeSpace fse;
	fse.initialize(_base, _imageWidth, _imageHeight);
	std::chrono::milliseconds::rep times = 0;
    std::cout << "[++] initialized" << std::endl;

	for (int n = 0; n < numData; n++)
	{
		char inpath[200];
        sprintf(inpath, imagePath, n+1);
        cv::Mat imgray = cv::imread(inpath, CV_LOAD_IMAGE_GRAYSCALE);
        sprintf(inpath, dispPath, n);
		// Load data1 (*.dat)
		cv::Mat imdisp = loadDisp(_imageHeight, _imageWidth, inpath);
		if (imdisp.empty() || imgray.empty()) return 0;

		// Estimate free space
		auto start = std::chrono::steady_clock::now();
		cv::Mat imfs = fse.detect(imdisp);
		auto end = std::chrono::steady_clock::now();
		auto diff = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);
		times += diff.count();

		// Draw the result;
		cv::Mat imdraw = fse.draw(imgray, imdisp, imfs);

		cv::imshow("result", imdraw);
		cv::waitKey(1000);

		// Save results
		char outpath[200];
        sprintf(outpath, resultPath, n);
		cv::imwrite(outpath, imdraw);
	}

	std::cout << "Average computational time : " << (double)times / (double)(numData) <<
		"ms" << std::endl;
}