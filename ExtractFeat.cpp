#include "ExtractFeat.h"

using namespace cv;
using namespace std;

//----------------------------------------------Uden-for-loop--------------------------------------------------------------------------------------------------
void ExtractFeat::clearFileContent() 
{
	ofstream ofs;
	ofs.open(data_file_path, std::ofstream::out | std::ofstream::trunc);										 // Open and clear content
	ofs << "Name,Width,Height,Area,Hist_mean blue,Hist_mean green,Hist_mean red,Bloodstains,Indents,Hullarity\n";
	ofs.close();
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
			if (color_arr[0].at<uchar>(y, x) >(color_arr[2].at<uchar>(y, x)))
				if (color_arr[0].at<uchar>(y, x) >(color_arr[1].at<uchar>(y, x)))
					bin.at<uchar>(y, x) = 0;
		}
	}
}

//----------------------------------------------Nuv�rende-fisk--------------------------------------------------------------------------------------------------

void ExtractFeat::getMeanHist(Fillet &fillet) 
{
	vector<Mat> bgr_planes; 	/// Separate the image in 3 planes ( B, G and R )
	split(fillet.img, bgr_planes);

	int histSize = 256; 	/// Establish the number of bins
	float range[] = { 0, 256 };		/// Set the ranges ( for B,G,R) )
	const float* histRange = { range };

	bool uniform = true, accumulate = false;

	Mat b_hist, g_hist, r_hist;

	/// Compute the histograms:
	calcHist(&bgr_planes[0], 1, nullptr, Mat(), b_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[1], 1, nullptr, Mat(), g_hist, 1, &histSize, &histRange, uniform, accumulate);
	calcHist(&bgr_planes[2], 1, nullptr, Mat(), r_hist, 1, &histSize, &histRange, uniform, accumulate);

	/// Compute the weighted mean
	float sum[3] = { 0 };
	int pixels[3] = { 0 };
	
	for (int i = 0; i < b_hist.rows; i++)										// Start at i = 0 to ignore black pixels
	{				
	
		pixels[0] += b_hist.at<float>(i);
		pixels[1] += g_hist.at<float>(i);
		pixels[2] += r_hist.at<float>(i);

		sum[0] += b_hist.at<float>(i)*i;
		sum[1] += g_hist.at<float>(i)*i;
		sum[2] += r_hist.at<float>(i)*i;
	}

	fillet.hist_mean[0] = sum[0] / pixels[0];
	fillet.hist_mean[1] = sum[1] / pixels[1];
	fillet.hist_mean[2] = sum[2] / pixels[2];
}

void ExtractFeat::getDimensions(Fillet &fillet) 
{
	const RotatedRect minR = minAreaRect(fillet.contour); // Define the minimum rectangle around the contour

	fillet.area = contourArea(fillet.contour); // Define the minimum rectangle around the contour

	if (minR.size.width < minR.size.height) // Make sure "width" is the smallest value
	{
		fillet.width = minR.size.width;
		fillet.height = minR.size.height;
	}
	else 
	{
		fillet.width = minR.size.height;
		fillet.height = minR.size.width;
	}
	/// Get the moments
	Moments mu;

		mu = moments(fillet.contour, false);


	///  Get the mass centers with use of moment

		fillet.contour_center_mass = Point2f(mu.m10 / mu.m00, mu.m01 / mu.m00);

		drawMarker(fillet.img, fillet.contour_center_mass, Scalar(0, 255, 0), MARKER_CROSS, 10, 1);

		// Calculate the center point compared to the original img
		Point center = Point(int(minR.center.x),int(minR.center.y));

		// Draw the minRect center
		drawMarker(fillet.img, center, Scalar(0, 0, 255), MARKER_CROSS, 10, 1);

}

void ExtractFeat::getBloodStains(Fillet &fillet) 
{
	Mat color_arr[3];
	split(fillet.img, color_arr);										// spilt image into 3 channels

	Mat bin = color_arr[2];												// Using the red(R) color space 'bin'

	threshold(bin, bin, 130, 255, 0);									// Threshold to find dark spots
	medianBlur(bin, bin, 21); 											// Blurring to get a more smooth BLOB


	// Blob detection
	SimpleBlobDetector::Params params;

	// Filter by color, 255 for white pixels
	params.filterByColor = true;
	params.blobColor = 0;

	// Filter by Area (Min and max to only get pixels defining a common blood stain)
	params.filterByArea = true;
	params.minArea = 1600; //bloodspot minmum st�rrelse
	params.maxArea = 6400; //bloodspot maximuum st�rrelse

	// Filter by Inertia
	params.filterByInertia = true;
	params.minInertiaRatio = 0.2; //hvor meget bloodspottet afviger sig fra en cirkel

	Ptr<SimpleBlobDetector> detector = SimpleBlobDetector::create(params);
	vector<KeyPoint> keypoints;
	detector->detect(bin, keypoints);

	// Contour detection (PART 2)
	vector<vector<Point> > contours;

	findContours(bin, contours, CV_RETR_TREE, CV_CHAIN_APPROX_SIMPLE);	// Find contours
	 
	// Approximate contours to get bounding rects
	vector<vector<Point>> contours_poly(contours.size());

	
	for (int i = 0; i < contours.size(); i++) // Draw contours
	{
		if (contourArea(contours[i]) < params.minArea || contourArea(contours[i]) > params.maxArea) 		// Skip if area is too small or too large
		{ 
			continue;
		}
		approxPolyDP(Mat(contours[i]), contours_poly[i], 3, true);											// Approximate contours to get bounding rects
		Rect boundRect;
		boundRect = boundingRect(Mat(contours_poly[i]));

		for (auto &keypoint : keypoints) 
		{
			if (boundRect.contains(keypoint.pt))																// Only draw contours that contains a keypoint
			{
				fillet.bloodstain_contours.push_back(contours_poly[i]);											// Save the bloodstain contour
				break;																							// Break out of current for-loop to go to next contour
			}
		}

	}
}

void ExtractFeat::getIndents(Fillet &fillet)
{
	int edgeSize = 50;
	vector<vector<Point>> indent_contour;
	
	Mat bin_region = Mat(fillet.bin.rows+ edgeSize*2, fillet.bin.cols+edgeSize*2 , CV_8U, Scalar(0,0,0));
	Mat bin_indent = Mat(fillet.bin.rows+ edgeSize*2, fillet.bin.cols+ edgeSize*2, CV_8U);
	Rect region = Rect(edgeSize, edgeSize, fillet.boundRect.width, fillet.boundRect.height);


	fillet.bin.copyTo(bin_region(region));


	Mat element = getStructuringElement(MORPH_RECT, Size(30, 30)); 

	//morphologyEx(bin, bin_indent, MORPH_OPEN, element); //opening
	morphologyEx(bin_region, bin_indent, MORPH_CLOSE, element); //closing
	
	absdiff(bin_indent,bin_region, bin_indent); //take the absulut value of difference and saves it in bin_indent

	Mat element2 = getStructuringElement(MORPH_RECT, Size(3, 3));

	morphologyEx(bin_indent, bin_indent, MORPH_OPEN, element2); //closing

	vector<vector<Point>> indent_contours;						// Find the contours on the binary image
	findContours(bin_indent(region), indent_contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);
	for (int i = 0; i < indent_contours.size(); i++)
	{
		if (contourArea(indent_contours[i]) > 150)
		{
			fillet.indendts.push_back(indent_contours[i]);
		}
		
	}
}

void ExtractFeat::getShape(Fillet &fillet)
{
	
	

	/// Find the convex hull object for each contour
	vector<Point> hull(fillet.contour);

		convexHull((fillet.contour), hull, false);
		fillet.hullarity = ((contourArea(fillet.contour))/ contourArea(hull));

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
	datafile << fillet.width << ',' << fillet.height << ',' << fillet.area << ',';
	datafile << fillet.hist_mean[0] << ',' << fillet.hist_mean[1] << ',' << fillet.hist_mean[2] << ',';
	datafile << fillet.bloodstain_contours.size() << ',';
	datafile << fillet.indendts.size() << ',';
	datafile << fillet.hullarity << '\n';
	datafile.close();
}

void ExtractFeat::drawFeatures(Mat &img, const Fillet &fillet) 
{
	// Define the minimum Rect (again)
	const RotatedRect minR = minAreaRect(fillet.contour);

	// Draw contour outline
	polylines(img(fillet.boundRect), fillet.contour, true, Scalar(255, 255, 255), 1);

	// Draw the rotated minRect
	Point2f rect_points[4];
	minR.points(rect_points);
	for (int j = 0; j < 4; j++) {
		// Shift the points by boundingRect start coordinates
		Point p1 = Point(rect_points[j].x + fillet.boundRect.x, rect_points[j].y + fillet.boundRect.y);
		Point p2 = Point(rect_points[(j + 1) % 4].x + fillet.boundRect.x, rect_points[(j + 1) % 4].y + fillet.boundRect.y);
		line(img, p1, p2, Scalar(0, 255, 0), 1, 8);
	}

	// Calculate the center point compared to the original img
	Point center = Point(fillet.boundRect.x + int(fillet.contour_center_mass.x), fillet.boundRect.y + int(fillet.contour_center_mass.y));

	// Draw the minRect center
	//drawMarker(img, center, Scalar(0, 255, 0), MARKER_CROSS, 10, 1);
	

	//draw Center of mass of the fillet.
	drawMarker(img, center, Scalar(0, 255, 0), MARKER_CROSS, 10, 1);
	circle(img, center, 6, Scalar(0, 255, 0));

	// Put fish name on image
	putText(img, fillet.name, Point(center.x - 60, center.y - 10), FONT_HERSHEY_COMPLEX_SMALL, 1, cvScalar(200, 200, 250), 1, CV_AA);

	// Draw poly around the bloodstain
	drawContours(img(fillet.boundRect), fillet.bloodstain_contours, -1, Scalar(0, 0, 255), 2);

	// Draw fill the indents
	drawContours(img(fillet.boundRect), fillet.indendts, -1, Scalar(0, 0, 255),-1);

	// Draw a circle around the indents
	/*for (auto &indent : fillet.indendts) {
		// Add the coordinates of the boundingRect starting point
		Point indent_center = (indent + fillet.boundRect.tl());
		circle(img, indent_center, 10, Scalar(0, 0, 255), 3);
	}*/
}
//----------------------------------------------MAIN--------------------------------------------------------------------------------------------------
void ExtractFeat::run(vector<Mat> &images)
{
	clearFileContent(); // Clear content of file to write new data


	for (int index = 0; index < images.size(); index++)								// Loop through all the images
	{
		
		Mat bin = Mat(images[index].rows, images[index].cols, CV_8U, 255);

		makeBinary(images[index], bin);						// Make the image binary (black with white spots)

		medianBlur(bin, bin, 7);							// Reduce salt-and-peper noise


		vector<vector<Point>> contours;						// Find the contours on the binary image
		findContours(bin, contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_SIMPLE);

		int fishNumber = 0;
		for (int i = 0; i < contours.size(); i++)
		{

			// Skip if the area is too small
			if (contourArea(contours[i]) < 10000)
			{
			continue;
			}
			fishNumber++;
			///////////////// START PROCESSING /////////////////
			Fillet new_fillet;

			



			
			//drawContours(mask, contours, i, Scalar(255, 255, 255), -1);

			// Save the contour points in relation to the bounding box
			new_fillet.boundRect = boundingRect(contours[i]);

			//Mat mask = Mat(images[index].rows, images[index].cols, CV_8UC3,
			new_fillet.bin = Mat(new_fillet.boundRect.height, new_fillet.boundRect.width, CV_8U, Scalar(0, 0, 0));
			new_fillet.img = Mat(new_fillet.boundRect.height, new_fillet.boundRect.width, CV_8UC3, Scalar(0, 0, 0));

			for (auto &contour_point : contours[i]) {
				new_fillet.contour.emplace_back(contour_point.x - new_fillet.boundRect.x, contour_point.y - new_fillet.boundRect.y);
			}


		
			

			vector<vector<Point>> current_contour;

			current_contour.push_back(new_fillet.contour);

			//hvorn�r bliveer det her gjordt?
			drawContours(new_fillet.bin, current_contour, 0, Scalar(255, 255, 255), -1); // tegner contour af den fisk der er igang i loopet. 

			
			// Copy only where the boundingRect is
			images[index](new_fillet.boundRect).copyTo(new_fillet.img, bin(new_fillet.boundRect));


			
			

			// Generate name of the individual countours
			// fish [index of image]-[index of contour] TODO: Use timestamp?
			new_fillet.name = "fish " + to_string(index) + "-" + to_string(fishNumber);
			
			// Calculates the mean histogram value of each BGR channel
			
			getMeanHist(new_fillet);

			getDimensions(new_fillet);														// Saves desired dimentions of contour to Fillet object
																				
			getBloodStains(new_fillet);														// Detects bloodstains and draws them on input image

			getIndents(new_fillet);															//Detects indents and gives coordinates of them.

			getShape(new_fillet);
			// -------after-Features-------------------------------------
			saveFeatures(new_fillet);

			drawFeatures(images[index], new_fillet);

			//displayImg(new_fillet.name, new_fillet.img);

			//displayImg("hello" + new_fillet.name, new_fillet.bin);
		}
		
		String name_orig = "Original img " + to_string(index);
		displayImg(name_orig, images[index]);
		//String name_bin = "Binary img " + to_string(index);
		//displayImg(name_bin, bin);
	}
	waitKey(0);
}


