/**
 * Program to detect pupil, based on
 * http://www.codeproject.com/Articles/137623/Pupil-or-Eyeball-Detection-and-Extraction-by-C-fro
 * with some improvements.
 */
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <cmath>
#include <iostream>
#include <stdio.h>


using namespace std;
using namespace cv;

RNG rng(12345);

int main(int argc, char** argv)
{
	double largest = 0; // Largest ellipse area
	int indx = 0; //Index of largest ellipse area
	
	// Load image
	Mat src = imread("eye1.jpg");
	if (src.empty())
		return -1;

	Mat srccopy = src.clone();
	// Invert the source image and convert to grayscale
	Mat gray;
	cvtColor(~src, gray, CV_BGR2GRAY);

	imshow("work 1", gray);
	Mat graycopy = gray.clone();

	// Convert to binary image by thresholding it
	threshold(gray, gray, 220, 255, THRESH_BINARY);
	imshow("work 2", gray);

	// Find all contours
	vector<vector<Point> > contours;
	findContours(gray.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

	cout << "Number of contours = " << contours.size() << endl;

	// Fill holes in each contour
	drawContours(gray, contours, -1, CV_RGB(255,255,255), -1);

	// Modification to look for ellipses
	vector<RotatedRect> minRect( contours.size() );
	vector<RotatedRect> minEllipse( contours.size() );

	for( int i = 0; i < contours.size(); i++ )
     { minRect[i] = minAreaRect( Mat(contours[i]) );
		cout << "Rect size = " << i << ": " << contours[i].size() << endl;
       if( contours[i].size() > 5 )
         { minEllipse[i] = fitEllipse( Mat(contours[i]) );
			cout << "Ellipse size = " << i << ": " << contours[i].size() << endl;
	   }
	}
	

	// Back to Normal
	for (int i = 0; i < contours.size(); i++)
	{
		double area = contourArea(contours[i]);
		cout << "Contour area = " << i << ": " << area << endl;
		if (area > largest){
			largest = area;
			indx = i;
		}
		Rect rect = boundingRect(contours[i]);
		int radius = rect.width/2;

		// If contour is big enough and has round shape
		// Then it is the pupil
		if (area >= 30 && 
		    abs(1 - ((double)rect.width / (double)rect.height)) <= 0.2 &&
				abs(1 - (area / (CV_PI * pow((double)radius, (double)2)))) <= 0.2)	
		{
			circle(src, Point(rect.x + radius, rect.y + radius), radius, CV_RGB(255,0,0), 2);
		}
	}

	// Mod to draw ellipses
//	for( int i = 0; i< contours.size(); i++ )
//     {
       // ellipse
       ellipse( graycopy, minEllipse[indx], CV_RGB(255,0,0), 2, 8 );
	   ellipse( srccopy, minEllipse[indx], CV_RGB(255,0,0), 2, 8 );
       // rotated rectangle
       Point2f rect_points[4]; minRect[indx].points( rect_points );
       for( int j = 0; j < 4; j++ ){
          line( graycopy, rect_points[j], rect_points[(j+1)%4], CV_RGB(0,255,0), 1, 8 );
		  line( srccopy, rect_points[j], rect_points[(j+1)%4], CV_RGB(0,255,0), 1, 8 );
	   }
   //  }
	imshow("work 3", graycopy);
	imshow("Original", srccopy);
	imshow("image", src);
	waitKey(0);

	return 0;
}
