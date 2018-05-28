#include "ExtractFeat.h"

using namespace cv;
using namespace std;

void ExtractFeat::clearFileContent()
{
    if (testingMode==false)
    {
        ofstream ofs;
        ofs.open(data_file_path, std::ofstream::out | std::ofstream::trunc); // Open and clear content
        ofs << "Name,Area,S_Mean,V_Mean,Bloodstains,Notches,Convexity,Skin_Area\n";
        ofs.close();
    }
    else
    {
        ofstream ofs;
        ofs.open(data_file_path_classification, std::ofstream::out | std::ofstream::trunc); // Open and clear content
        ofs.close();
    }
}

void ExtractFeat::displayImg(const String &name, const Mat &img)
{
    namedWindow(name, CV_WINDOW_NORMAL);
    resizeWindow(name, img.cols / 1, img.rows / 1);
    imshow(name, img);
}

void ExtractFeat::makeBinary(const Mat &img, Mat &bin)
{
    Mat color_arr[3];
    split(img, color_arr);

    for (int x = 0; x < img.cols; x++) {
        for (int y = 0; y < img.rows; y++) {
            // Check if blue pixel is "stronger" than red and green
            int blue = color_arr[0].at<uchar>(y, x);
            int green = color_arr[1].at<uchar>(y, x);
            int red = color_arr[2].at<uchar>(y, x);
            if ((blue > red) && (blue > green))
                bin.at<uchar>(y, x) = 0;
        }
    }

    medianBlur(bin, bin, 7); // Reduce salt-and-peper noise
}

//----------------------------------------------Nuværende-fisk--------------------------------------------------------------------------------------------------

void ExtractFeat::getMean(Fillet &fillet)
{
    Mat hsv_img;
    cvtColor(fillet.img, hsv_img, CV_BGR2HSV);

    Scalar means = mean(hsv_img, fillet.bin);

    fillet.hist_mean[0] = means.val[1];
    fillet.hist_mean[1] = means.val[2];
}


void ExtractFeat::getBloodstains(Fillet &fillet)
{
    Mat hsv_img;
    Mat hsv_arr[3];

    cvtColor(fillet.img, hsv_img, COLOR_BGR2HSV);

    split(hsv_img, hsv_arr);

    Mat s_channel = hsv_arr[1];
    Mat v_channel = hsv_arr[2];

    threshold(s_channel, s_channel, 165, 255, 0);

    threshold(v_channel, v_channel, 150, 255, 0);

    Mat mix = s_channel + v_channel;
    medianBlur(mix, mix, 29);

    vector<vector<Point> > contours_bin;
    findContours(mix, contours_bin, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);	// Find contours

    for (int i = 0; i < contours_bin.size(); i++)
    {
        double contArea = contourArea(contours_bin[i]);

        if (contArea < 2000 || contArea > 10000)
            continue;

        vector<Point> hull;
        convexHull(contours_bin[i], hull);

        if (contArea / contourArea(hull) < 0.91)
            continue;

        Mat mask = Mat(hsv_img.rows, hsv_img.cols, CV_8U, Scalar(0));
        drawContours(mask, contours_bin, i, 255, -1);
        Scalar means = mean(fillet.img, mask);

        if (means.val[2] > 130)
            continue;

        //The remaining contour contains a bloodstain.
        fillet.bloodstain = true;
    }
}

void ExtractFeat::getNotches(Fillet &fillet)
{
    int edgeSize = 50;
    vector<vector<Point>> notch_contour;

    Mat bin_region = Mat(fillet.bin.rows + edgeSize * 2, fillet.bin.cols + edgeSize * 2, CV_8U, Scalar(0, 0, 0));
    Mat bin_notch;

    Rect region = Rect(edgeSize, edgeSize, fillet.bin.cols, fillet.bin.rows);


    fillet.bin.copyTo(bin_region(region));


    Mat element = getStructuringElement(MORPH_RECT, Size(30, 30));

    morphologyEx(bin_region, bin_notch, MORPH_CLOSE, element); //closing


    absdiff(bin_notch, bin_region, bin_notch); //take the absulut value of difference and saves it in bin_notch


    Mat element2 = getStructuringElement(MORPH_RECT, Size(3, 3));

    morphologyEx(bin_notch, bin_notch, MORPH_OPEN, element2); //opening

    vector<vector<Point>> notch_contours;		// Find the contours on the binary image
    findContours(bin_notch(region), notch_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);



    for (int i = 0; i < notch_contours.size(); i++)
    {
        double notch_area = contourArea(notch_contours[i]);
        if (notch_area > fillet.largestNotch)
        {
            fillet.largestNotch = notch_area;
        }
    }
}

void ExtractFeat::getShape(Fillet &fillet)
{
    // Find the convex hull object for each contour
    vector<Point> hull;
    convexHull((fillet.contour), hull);
    fillet.convexity = ((contourArea(fillet.contour)) / contourArea(hull));
}

void ExtractFeat::getSkin(Fillet &fillet)
{
    int edgeSize = 50;

    Mat skin_region = Mat(fillet.img.rows + edgeSize * 2, fillet.img.cols + edgeSize * 2, CV_8U, Scalar(0, 0, 0));

    Rect region = Rect(edgeSize, edgeSize, fillet.img.cols, fillet.img.rows);

    Mat skinimg;
    cvtColor(fillet.img, skinimg, COLOR_BGR2HSV);

    vector<Mat> hsv_planes; // Separate the image in 3 planes ( H, S, and V )
    split(skinimg, hsv_planes);

    hsv_planes[1].copyTo(skin_region(region));

    // Set threshold and maxValue
    double thresh = 125;
    double maxValue = 255;
    // Binary Threshold
    medianBlur(skin_region, skin_region, 21);
    threshold(skin_region, skin_region, thresh, maxValue, THRESH_TOZERO_INV);
    threshold(skin_region, skin_region, 1, maxValue, THRESH_BINARY);

    Mat element = getStructuringElement(MORPH_RECT, Size(23, 23));

    morphologyEx(skin_region, skin_region, MORPH_OPEN, element);

    vector<vector<Point>> skin_contour;

    findContours(skin_region(region), skin_contour, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
    for (int i = 0; i< skin_contour.size(); i++)
    {
        fillet.skinArea += contourArea(skin_contour[i]);
    }
}

//-------------------------------------------Efter-Features--------------------------------------------------------------------------------------------------------------------
/**
* Saves the features to a file.
* @param [in] fillet - Reference object
*/
void ExtractFeat::saveFeatures(const Fillet &fillet)
{
    ofstream datafile;
    datafile.open(data_file_path, std::ios_base::app);

    datafile << fillet.name << ',';
    datafile << fillet.area << ',';
    datafile << fillet.hist_mean[0] << ',' << fillet.hist_mean[1] << ',';
    datafile << fillet.bloodstain << ',';
    datafile << fillet.largestNotch << ',';
    datafile << fillet.convexity << ',';
    datafile << fillet.skinArea << '\n';
    datafile.close();
}

void ExtractFeat::Classify(const Fillet &fillet)
{
    ofstream datafile;
    datafile.open(data_file_path_classification, std::ios_base::app);

    datafile << fillet.name << ',';

    datafile << fillet.classification << ',';
    datafile << fillet.reason << '\n';
    datafile.close();
}

//----------------------------------------------Testing-------------------------------------------------------------------------------------------------
void ExtractFeat::runTesting(vector<Mat> &images)
{
    clearFileContent(); // Clear content of file to write new data

    for (int index = 0; index < images.size(); index++)			// Loop through all the images
    {

        Mat bin = Mat(images[index].rows, images[index].cols, CV_8U, 255);

        makeBinary(images[index], bin);		// Make the image binary (black with white spots)

        vector<vector<Point>> contours;		// Find the contours on the binary image
        findContours(bin, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        int filletCounter = 0;
        for (int i = 0; i < contours.size(); i++)
        {
            double contour_area = contourArea(contours[i]);
            // Skip if the area is too small
            if (contour_area < 30000)
            {
                continue;
            }

            filletCounter++;

            ///////////////// START PROCESSING /////////////////
            Fillet new_fillet;
            new_fillet.area = contour_area;

            // Save the contour points in relation to the bounding box
            new_fillet.boundRect = boundingRect(contours[i]);

            //Mat mask = Mat(images[index].rows, images[index].cols, CV_8UC3,
            new_fillet.bin = Mat(new_fillet.boundRect.height, new_fillet.boundRect.width, CV_8U, Scalar(0));
            new_fillet.img = Mat(new_fillet.boundRect.height, new_fillet.boundRect.width, CV_8UC3, Scalar(0, 0, 0));

            for (int j = 0; j < contours[i].size(); j++) {
                new_fillet.contour.emplace_back(contours[i][j].x - new_fillet.boundRect.x, contours[i][j].y - new_fillet.boundRect.y);
            }

            vector<vector<Point>> current_contour;

            current_contour.push_back(new_fillet.contour);

            // Saves a mask of the current fillet.
            drawContours(new_fillet.bin, current_contour, 0, Scalar(255, 255, 255), -1);

            // Copy only where the boundingRect is
            images[index](new_fillet.boundRect).copyTo(new_fillet.img, bin(new_fillet.boundRect));

            // Generate name of the individual countours
            // fish [index of image]-[index of contour]
            string index_str = (index > 9) ? to_string(index) : "0" + to_string(index);
            new_fillet.name = "fish-" + index_str + "-" + to_string(filletCounter);

            int m = 10;
            int b = 1500;
            getMean(new_fillet); // Calculates the mean histogram value of each HSV channel
            if (m*new_fillet.hist_mean[0] + new_fillet.hist_mean[1] >= b) // meat side
            {// if over the b value it is the meat side.
                getBloodstains(new_fillet);														// Detects bloodstains and draws them on input image
                if (new_fillet.bloodstain == true)
                {
                    new_fillet.classification = 1;
                    new_fillet.reason = "Bloodstain";
                }
                else
                {
                    getShape(new_fillet);
                    if (new_fillet.convexity<0.9436)
                    {
                        new_fillet.classification = 1;
                        new_fillet.reason = "Deformity";
                    }
                    else
                    {
                        getNotches(new_fillet);															//Detects notches and gives coordinates of them.
                        if (new_fillet.largestNotch>142.5)
                        {
                            new_fillet.classification = 1;
                            new_fillet.reason = "Notch";
                        }

                        else
                            new_fillet.classification = 3;
                    }
                }
            }
            else //Skin side
            {
                // if over the b value it is the meat side.
                getSkin(new_fillet);														// Detects bloodstains and draws them on input image
                if (new_fillet.skinArea>66000)
                {
                    new_fillet.classification = 2;
                    new_fillet.reason = "Excessive Skin";
                }
                else
                {
                    getShape(new_fillet);
                    if (new_fillet.convexity<0.9436)
                    {
                        new_fillet.classification = 2;
                        new_fillet.reason = "Deformity";
                    }
                    else
                    {
                        getNotches(new_fillet);															//Detects notches and gives coordinates of them.
                        if (new_fillet.largestNotch > 142.5)
                        {
                            new_fillet.classification = 2;
                            new_fillet.reason = "Notch";
                        }
                        else
                            new_fillet.classification = 4;
                    }
                }
            }
            // -------After binary tree-------------------------------------
            Classify(new_fillet);

        }

        String name_orig = "Original img " + to_string(index);
        displayImg(name_orig, images[index]);

    }

    waitKey(0);

}
//----------------------------------------------TRAINING--------------------------------------------------------------------------------------------------
void ExtractFeat::runTraining(vector<Mat> &images)
{
    clearFileContent(); // Clear content of file to write new data

    for (int index = 0; index < images.size(); index++)			// Loop through all the images
    {

        Mat bin = Mat(images[index].rows, images[index].cols, CV_8U, 255);

        makeBinary(images[index], bin);		// Make the image binary (black with white spots)

        vector<vector<Point>> contours;		// Find the contours on the binary image
        findContours(bin, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

        int filletCounter = 0;
        for (int i = 0; i < contours.size(); i++)
        {
            double contour_area = contourArea(contours[i]);
            // Skip if the area is too small
            if (contour_area < 30000)
            {
                continue;
            }

            filletCounter++;

            ///////////////// START PROCESSING /////////////////
            Fillet new_fillet;
            new_fillet.area = contour_area;

            // Save the contour points in relation to the bounding box
            new_fillet.boundRect = boundingRect(contours[i]);

            //Mat mask = Mat(images[index].rows, images[index].cols, CV_8UC3,
            new_fillet.bin = Mat(new_fillet.boundRect.height, new_fillet.boundRect.width, CV_8U, Scalar(0));
            new_fillet.img = Mat(new_fillet.boundRect.height, new_fillet.boundRect.width, CV_8UC3, Scalar(0, 0, 0));

            for (int j = 0; j < contours[i].size(); j++) {
                int relativeX = contours[i][j].x - new_fillet.boundRect.x;
                int relativeY = contours[i][j].y - new_fillet.boundRect.y;

                new_fillet.contour.emplace_back(relativeX, relativeY);

            }

            // Saves a mask of the current fillet.
            drawContours(new_fillet.bin, vector<vector<Point>> (1, new_fillet.contour), 0, 255, -1);

            // Copy only where the boundingRect is
            images[index](new_fillet.boundRect).copyTo(new_fillet.img, new_fillet.bin);

            // Generate name of the individual countours
            // fish [index of image]-[index of contour]
            string index_str = (index > 9) ? to_string(index) : "0" + to_string(index);
            new_fillet.name = "fish-" + index_str + "-" + to_string(filletCounter);

            //TRAINING-----------------------------------------------------------
            getMean(new_fillet); // Calculates the mean histogram value of each HSV channel

            getBloodstains(new_fillet);														// Detects bloodstains and draws them on input image

            getNotches(new_fillet);															//Detects notches and gives coordinates of them.

            getShape(new_fillet);

            getSkin(new_fillet);

            // -------after-Features-------------------------------------
            saveFeatures(new_fillet);

            //displayImg("fillet.img_" + new_fillet.name, new_fillet.img);

        }

        String name_orig = "Original img " + to_string(index);
        displayImg(name_orig, images[index]);

    }

    waitKey(0);

}

