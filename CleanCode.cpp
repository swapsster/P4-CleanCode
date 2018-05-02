#include "ExtractFeat.h"
#include "stdafx.h"

#include <cstdio>
#include <opencv2/core.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/opencv.hpp>
#include <iostream> // cout
#include <fstream> // Open files
#include <math.h>

using namespace cv;
using namespace std;

void loadImages(const String &path, vector<Mat> &images)
{
	vector<String> fn;
	glob(path, fn, true); // recurse
	for (const auto &k : fn) {
		Mat im = imread(k);
		if (im.empty()) continue; //only proceed if sucsessful
		  
		im = im(Rect(0, 330, 1936, 1037 - 330)); // This Rect is approx only the conveyor

		images.push_back(im);
	}
}

int main()
{
	// Vector for all images to be proccesed
	vector<Mat> images;
	//"../resources/images/good/*.tif"
	loadImages("C:\\Users\\Swapsster\\Documents\\p4\\project\\CleanCode\\Good\\*.tif", images);

	ExtractFeat classifier;
	classifier.run(images);

	return 0;
}