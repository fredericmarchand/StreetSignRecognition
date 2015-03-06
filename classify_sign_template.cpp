#include "opencv2/core/core.hpp"
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include "opencv2/calib3d/calib3d.hpp"
#include <iostream>
#include <stdio.h>
#include <stdlib.h>

#define WARPED_XSIZE 200
#define WARPED_YSIZE 300
#define DEBUG 0

using namespace cv;
using namespace std;

Mat src; Mat src_gray, warped_result;
Mat speed_80, speed_40;
int canny_thresh = 154;

#define VERY_LARGE_VALUE 100000

#define NO_CLASSIFICATION    0
#define STOP_SIGN            1
#define SPEED_LIMIT_40_SIGN  2
#define SPEED_LIMIT_80_SIGN  3

/** @function main */
int main( int argc, char** argv)
{
	int sign_recog_result = NO_CLASSIFICATION;
	speed_40 = imread("speed_40.bmp", 0);
	speed_80 = imread("speed_80.bmp", 0);

	// you run your program on these three examples (uncomment the two lines below)
	string sign_name = "stop4";
	//string sign_name = "speedsign12";
	//string sign_name = "speedsign3";
	string final_sign_input_name = sign_name + ".jpg";
	string final_sign_output_name = sign_name + "_result" + ".jpg";

	/// Load source image and convert it to gray
	src = imread (final_sign_input_name, 1);

	/// Convert image to gray and blur it
	cvtColor(src, src_gray, COLOR_BGR2GRAY);
	blur(src_gray, src_gray, Size(3,3));
	warped_result = Mat(Size(WARPED_XSIZE, WARPED_YSIZE), src_gray.type());

	// here you add the code to do the recognition, and set the variable 
	// sign_recog_result to one of STOP_SIGN, SPEED_LIMIT_40_SIGN, or SPEED_LIMIT_80_SIGN

	Mat canny_output;
	vector<vector<Point> > contours;
	vector<Vec4i> hierarchy;
	RNG rng(12345);
	vector<vector<Point2f> > contoursOut;

	Canny(src_gray, canny_output, canny_thresh, canny_thresh*2, 3); //Finds edges in an image using the [Canny86] algorithm.
	findContours(canny_output, contours, hierarchy, RETR_EXTERNAL, CHAIN_APPROX_NONE, Point(0, 0)); //Finds contours in a binary image.

	contoursOut.resize(contours.size());
	for (int i = 0; i < contours.size(); ++i) 
	{
		approxPolyDP(Mat(contours[i]), contoursOut[i], contours[i].size()*.05, true); //Approximates a polygonal curve(s) with the specified precision.
	}
	
#if DEBUG == 1
	for (int i = 0; i < contoursOut.size(); ++i)
		cout << contoursOut[i] << endl;
#endif
	
	/// Draw contours
#if DEBUG == 1
	Mat drawing = Mat::zeros(canny_output.size(), CV_8UC3);
	for (int i = 0; i < contours.size(); i++)
	{
		Scalar color = Scalar(rng.uniform(0, 255), rng.uniform(0,255), rng.uniform(0,255));
		drawContours(drawing, contours, i, color, 2, 8, hierarchy, 0, Point());
	}

	/// Show in a window
	namedWindow("Contours", WINDOW_AUTOSIZE);
	imshow("Contours", drawing);
	waitKey(0);
#endif
	
	//Find out the biggest contour on the image
	int index = 0;
	double area = 0;
	for (int i = 0; i < contours.size(); ++i)
	{
		if (contourArea(contours[i]) > area)
		{
			index = i;
			area = contourArea(contours[i]);
		}
	}

	//if the biggest contour has 8 sides
	if (contoursOut[index].size() == 8) 
	{

#if DEBUG == 1
		for (int j = 0; j < contoursOut.size(); ++j) 
		{
			if (contoursOut[j].size() == 8) 
			{
				Point2f *pt[8];
				for(int i = 0; i < 8; i++) 
				{
					pt[i] = &contoursOut[j][i];
				}
   
				//drawing lines around the heptagon
				line(src, *pt[0], *pt[1], Scalar(0,0,255),4);
				line(src, *pt[1], *pt[2], Scalar(0,0,255),4);
				line(src, *pt[2], *pt[3], Scalar(0,0,255),4);
				line(src, *pt[3], *pt[4], Scalar(0,0,255),4);
				line(src, *pt[4], *pt[5], Scalar(0,0,255),4);
				line(src, *pt[5], *pt[6], Scalar(0,0,255),4);
				line(src, *pt[6], *pt[7], Scalar(0,0,255),4);
				line(src, *pt[7], *pt[0], Scalar(0,0,255),4);
			}
		}
#endif

		sign_recog_result = STOP_SIGN;
	}
	else
	{
		Mat dst;
		vector<Point2f> output(4);
		output[0] = Point(0, 0);
		output[1] = Point(0, speed_40.rows);
		output[2] = Point(speed_40.cols, speed_40.rows);
		output[3] = Point(speed_40.cols, 0);
		
		for (int j = 1; j <= 4; ++j) {
			std::rotate(contoursOut[index].begin(), contoursOut[index].begin()+1, contoursOut[index].end());

			Mat M = getPerspectiveTransform(contoursOut[index], output); //Calculates a perspective transform from four pairs of the corresponding points.
			warpPerspective(src_gray, dst, M, speed_40.size()); //Applies a perspective transformation to an image.
			
			Mat diff40 = dst != speed_40;
			Mat diff80 = dst != speed_80;

#if DEBUG == 1
			char* source_window = "Result";
			namedWindow(source_window, WINDOW_AUTOSIZE);
			imshow(source_window, dst);
			waitKey(0);	
#endif

			if (countNonZero(diff80) == 0)
			{
				sign_recog_result = SPEED_LIMIT_80_SIGN;
				break;
			}

			if (countNonZero(diff40) == 0)
			{
				sign_recog_result = SPEED_LIMIT_40_SIGN;
				break;
			}
		}
	}

	string text;
	if (sign_recog_result == SPEED_LIMIT_40_SIGN) text = "Speed 40";
	else if (sign_recog_result == SPEED_LIMIT_80_SIGN) text = "Speed 80";
	else if (sign_recog_result == STOP_SIGN) text = "Stop";
	else if (sign_recog_result == NO_CLASSIFICATION) text = "No Result";

	int fontFace = FONT_HERSHEY_SCRIPT_SIMPLEX;
	double fontScale = 2;
	int thickness = 3;  
	cv::Point textOrg(10, 130);
	cv::putText(src, text, textOrg, fontFace, fontScale, Scalar::all(255), thickness,8);

	/// Create Window
	char* source_window = "Result";
	namedWindow(source_window, WINDOW_AUTOSIZE);
	imshow(source_window, src);
	imwrite(final_sign_output_name, src);
	
	waitKey(0);

    return(0);
}
