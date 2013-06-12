#include "findPupil.h"

RotatedRect findPupil(Mat &coloredEye, Rect &eyeArea)
{

// Find the Pupils
		double largest = 0; // Largest ellipse area
		int indx = 0; //Index of largest ellipse area
		cout << "Eye Rect " << eyeArea << endl;
		imshow("Find Pupil", coloredEye);
		Mat gray;
		// Invert the source image and convert to grayscale
		cvtColor(~coloredEye, gray, CV_BGR2GRAY);
		imshow("Converted Pupil", gray);
		// Convert to binary image by thresholding it
		threshold(gray, gray, 220, 255, THRESH_BINARY);
		imshow("Thresh Pupil", gray);
		// Find all contours
		vector<vector<Point> > contours;
		findContours(gray.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

		cout << "Number of contours = " << contours.size() << endl;

		// Fill holes in each contour
		drawContours(gray, contours, -1, CV_RGB(255,255,255), -1);
		imshow("Contours Left Pupil", gray);

		// Modification to look for ellipses
		vector<RotatedRect> minRect( contours.size() );
		vector<RotatedRect> minEllipse( contours.size() );

		for( int i = 0; i < contours.size(); i++ )
			{ minRect[i] = minAreaRect( Mat(contours[i]) );
			cout << "Rect size = " << i << ": " << contours[i].size() << endl;
			  if( contours[i].size() > 5 )
				{ minEllipse[i] = fitEllipse( Mat(contours[i]) );
					cout << "Ellipse size = " << i << ": " << contours[i].size() << endl;
					cout << "Contour array " << Mat(contours[i]) << endl;
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
			}
		cout << "Largest area and index " << largest << ":" << indx << endl;
		Rect rect = boundingRect(contours[indx]);
		cout << "Rect = " << rect << endl;
		// ellipse
		if (largest>=5){
			ellipse(coloredEye, minEllipse[indx], CV_RGB(255,0,0), 1, 8 );
			//rectangle(LEye, rect, CV_RGB(255,0,0), 2, 8 );
			imshow("Found Pupil", coloredEye);
		}
	return minEllipse[indx];
}
