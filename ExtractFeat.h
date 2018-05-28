#pragma once

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

struct Fillet {
    String name;
    double hist_mean[2] = { 0 };						// Saturation, Value
    double area = 0;
    double convexity = 0;						// Contour area + convexity(squarity) which is contour area divided by boundrect area.
    double skinArea = 0;
    double largestNotch = 0;
    bool bloodstain = false;
    int classification = 0; 				//bad meat = 1; bad skin =2; good meat 3; good skin=4;
    string reason = "None";



    Rect boundRect;									// The img is generated from the original image using this boundingRect
    vector<Point> contour;							// Coordinates of the fillet
    Mat img, bin;									// Only the boundingRect image from original image + Binary image of fillet
};

class ExtractFeat
{
public:
    bool testingMode = false;
    String data_file_path = "../data/features.dat";
    String data_file_path_classification = "../data/classification.dat";

    //------------Uden-For-Loop----------------------
    void clearFileContent();
    void displayImg(const String &name, const Mat &img);
    void makeBinary(const Mat &img, Mat &bin);
    //------------Nuværende-fisk----------------------

    void Classify(const Fillet &fillet);

    void getMean(Fillet &fillet);
    void getBloodstains(Fillet &fillet);
    void getNotches(Fillet &fillet);
    void getShape(Fillet &fillet);
    void getSkin(Fillet &fillet);

    //------------Efter-Features-------------------------------
    void saveFeatures(const Fillet &fillet);
    //------------Main----------------------
    void runTesting(vector<Mat> &images);
    void runTraining(vector<Mat> &images);
};

