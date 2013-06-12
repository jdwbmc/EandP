// Cascade Classifier file, used for Face Detection.
const char *faceCascadeFilename = "lbpcascade_frontalface.xml";     // LBP face detector.
//const char *faceCascadeFilename = "haarcascade_frontalface_alt_tree.xml";  // Haar face detector.
//const char *eyeCascadeFilename1 = "haarcascade_lefteye_2splits.xml";   // Best eye detector for open-or-closed eyes.
//const char *eyeCascadeFilename2 = "haarcascade_righteye_2splits.xml";   // Best eye detector for open-or-closed eyes.
const char *eyeCascadeFilename1 = "haarcascade_mcs_lefteye.xml";       // Good eye detector for open-or-closed eyes.
const char *eyeCascadeFilename2 = "haarcascade_mcs_righteye.xml";       // Good eye detector for open-or-closed eyes.
//const char *eyeCascadeFilename1 = "haarcascade_eye.xml";               // Basic eye detector for open eyes only.
//const char *eyeCascadeFilename2 = "haarcascade_eye_tree_eyeglasses.xml"; // Basic eye detector for open eyes if they might wear glasses.


// Set the desired face dimensions. Note that "getPreprocessedFace()" will return a square face.
const int faceWidth = 70;
const int faceHeight = faceWidth; 

// Try to set the camera resolution. Note that this only works for some cameras on
// some computers and only for some drivers, so don't rely on it to work!
const int DESIRED_CAMERA_WIDTH = 640;
const int DESIRED_CAMERA_HEIGHT = 480;

const char *windowName = "WebcamFaceEyes";   // Name shown in the GUI window.
const int BORDER = 8;  // Border between GUI elements to the edge of the image.

const bool preprocessLeftAndRightSeparately = true;   // Preprocess left & right sides of the face separately, in case there is stronger light on one side.

// Include OpenCV's C++ Interface
 #include "opencv2/opencv.hpp"
 #include "opencv2/objdetect/objdetect.hpp"
 #include "opencv2/highgui/highgui.hpp"
 #include "opencv2/imgproc/imgproc.hpp"
// Include the rest of our code!
 #include "detectObject.h"       // Easily detect faces or eyes (using LBP or Haar Cascades).
 #include "preprocessFace.h"     // Easily preprocess face images, for face recognition.
 #include "ImageUtils.h"      // Shervin's handy OpenCV utility functions.
 #include "findPupil.h"  //Find the pupil in the eye frame


 #include <iostream>
 #include <stdio.h>
 #include <vector>
 #include <string>

 using namespace std;
 using namespace cv;

 #if !defined VK_ESCAPE
    #define VK_ESCAPE 0x1B      // Escape character (27)
#endif

 /** Function Headers */
 void detectAndDisplay( Mat frame );

 /** Global variables */
 RNG rng(12345);


// Load the face and 1 or 2 eye detection XML classifiers.
void initDetectors(CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2)
{
    // Load the Face Detection cascade classifier xml file.
    try {   // Surround the OpenCV call by a try/catch block so we can give a useful error message!
        faceCascade.load(faceCascadeFilename);
    } catch (cv::Exception &e) {}
    if ( faceCascade.empty() ) {
        cerr << "ERROR: Could not load Face Detection cascade classifier [" << faceCascadeFilename << "]!" << endl;
        cerr << "Copy the file from your OpenCV data folder (eg: 'C:\\OpenCV\\data\\lbpcascades') into this WebcamFaceRec folder." << endl;
        exit(1);
    }
    cout << "Loaded the Face Detection cascade classifier [" << faceCascadeFilename << "]." << endl;

    // Load the Eye Detection cascade classifier xml file.
    try {   // Surround the OpenCV call by a try/catch block so we can give a useful error message!
        eyeCascade1.load(eyeCascadeFilename1);
    } catch (cv::Exception &e) {}
    if ( eyeCascade1.empty() ) {
        cerr << "ERROR: Could not load 1st Eye Detection cascade classifier [" << eyeCascadeFilename1 << "]!" << endl;
        cerr << "Copy the file from your OpenCV data folder (eg: 'C:\\OpenCV\\data\\haarcascades') into this WebcamFaceRec folder." << endl;
        exit(1);
    }
    cout << "Loaded the 1st Eye Detection cascade classifier [" << eyeCascadeFilename1 << "]." << endl;

    // Load the Eye Detection cascade classifier xml file.
    try {   // Surround the OpenCV call by a try/catch block so we can give a useful error message!
        eyeCascade2.load(eyeCascadeFilename2);
    } catch (cv::Exception &e) {}
    if ( eyeCascade2.empty() ) {
        cerr << "Could not load 2nd Eye Detection cascade classifier [" << eyeCascadeFilename2 << "]." << endl;
        // Dont exit if the 2nd eye detector did not load, because we have the 1st eye detector at least.
        //exit(1);
    }
    else
        cout << "Loaded the 2nd Eye Detection cascade classifier [" << eyeCascadeFilename2 << "]." << endl;
}


// Get access to the webcam.
void initWebcam(VideoCapture &videoCapture, int cameraNumber)
{
    // Get access to the default camera.
    try {   // Surround the OpenCV call by a try/catch block so we can give a useful error message!
        videoCapture.open(cameraNumber);
    } catch (cv::Exception &e) {}
    if ( !videoCapture.isOpened() ) {
        cerr << "ERROR: Could not access the camera!" << endl;
        exit(1);
    }
    cout << "Loaded camera " << cameraNumber << "." << endl;
}

/*
void onMouse(int event, int x, int y, int, void*)
{
    // We only care about left-mouse clicks, not right-mouse clicks or mouse movement.
    if (event != CV_EVENT_LBUTTONDOWN)
        return;
}
*/

void recognizeUsingWebcam(VideoCapture &videoCapture, CascadeClassifier &faceCascade, CascadeClassifier &eyeCascade1, CascadeClassifier &eyeCascade2)
{
    Ptr<FaceRecognizer> model;
    vector<Mat> preprocessedFaces;
    vector<int> faceLabels;
    Mat old_prepreprocessedFace;
	Mat LEye, REye; //Peepare for pupil detection using eye rectangle areas
    double old_time = 0;

       // Run forever, until the user hits Escape to "break" out of this loop.
    while (true) {

        // Grab the next camera frame. Note that you can't modify camera frames.
        Mat cameraFrame;
        videoCapture >> cameraFrame;
        if( cameraFrame.empty() ) {
            cerr << "ERROR: Couldn't grab the next camera frame." << endl;
            exit(1);
        }

        // Get a copy of the camera frame that we can draw onto.
        Mat displayedFrame;
        cameraFrame.copyTo(displayedFrame);

        // Find a face and preprocess it to have a standard size and contrast & brightness.
        Rect faceRect;  // Position of detected face.
        Rect searchedLeftEye, searchedRightEye; // top-left and top-right regions of the face, where eyes were searched.
		Rect dispLeftEye, dispRightEye; // Display eye searched regios.  Must do offset
        Point leftEye, rightEye;    // Position of the detected eyes.
        Mat preprocessedFace = getPreprocessedFace(displayedFrame, faceWidth, faceCascade, eyeCascade1, eyeCascade2, preprocessLeftAndRightSeparately, &faceRect, &leftEye, &rightEye, &searchedLeftEye, &searchedRightEye);

        bool gotFaceAndEyes = false;
        if (preprocessedFace.data)
            gotFaceAndEyes = true;

        // Draw an anti-aliased rectangle around the detected face.
        if (faceRect.width > 0) {
            rectangle(displayedFrame, faceRect, CV_RGB(255, 255, 0), 2, CV_AA);
			cout << "FRect x= " << faceRect.x << " FRect y= "<< faceRect.y << " FR width = "<< faceRect.width << " FR height = "<< faceRect.height << endl;

			// Draw light-blue anti-aliased circles for the 2 eyes.
            Scalar eyeColor = CV_RGB(0,255,255);
            if (leftEye.x >= 0) {   // Check if the eye was detected
				dispLeftEye = searchedLeftEye;
				dispLeftEye.x += faceRect.x;
				dispLeftEye.y += faceRect.y;
				Mat lfme = displayedFrame(dispLeftEye);
				LEye = lfme.clone();
				rectangle(displayedFrame, dispLeftEye, CV_RGB(255,0,255), 2, CV_AA);
                circle(displayedFrame, Point(faceRect.x + leftEye.x, faceRect.y + leftEye.y), 6, eyeColor, 1, CV_AA);
				cout << "LER x = " << searchedLeftEye.x << " LER y = " << searchedLeftEye.y << " LER width = " << searchedLeftEye.width << " LER height = " << searchedLeftEye.height << endl;
				cout << "LE x = " << leftEye.x << " LE y = " << leftEye.y << endl; 
            }
            if (rightEye.x >= 0) {   // Check if the eye was detected
				dispRightEye = searchedRightEye;
				dispRightEye.x += faceRect.x;
				dispRightEye.y += faceRect.y;
				Mat rfme = displayedFrame(dispRightEye);
				REye = rfme.clone();
				rectangle(displayedFrame, dispRightEye, CV_RGB(0,255,0), 2, CV_AA);
                circle(displayedFrame, Point(faceRect.x + rightEye.x, faceRect.y + rightEye.y), 6, eyeColor, 1, CV_AA);
				cout << "RER x = " << searchedRightEye.x << " RER y = " << searchedRightEye.y << " RER width = " << searchedRightEye.width << " RER height = " << searchedRightEye.height << endl;
				cout << "RE x = " << rightEye.x << " RE y = " << rightEye.y << endl; 
            }
        }

		if(leftEye.x > 0){
			RotatedRect rectL = findPupil(LEye, dispLeftEye);
			Rect rr = rectL.boundingRect();
			cout << "rrL = " << rr << endl;
			if (rr.size() > 5)
				ellipse(LEye, rectL, CV_RGB(255,0,0), 1, 8 );
			imshow("Ellipse L Eye", LEye);
		}
		if(rightEye.x > 0){
			RotatedRect rectL = findPupil(REye, dispRightEye);
			Rect rr = rectL.boundingRect();
			cout << "rrR = " << rr << endl;
			if (rr.size() > 5)
				ellipse(REye, rectL, CV_RGB(255,0,0), 1, 8 );
			imshow("Ellipse R Eye", REye);
		}
		
		/*
		// Find the Pupils
		double largest = 0; // Largest ellipse area
		double largest2 = 0;
		int indx = 0; //Index of largest ellipse area
		int indx2 = 0;
		if (leftEye.x > 0){
			cout << "Left Eye Rect " << dispLeftEye << endl;
			//Mat LEye(displayedFrame, dispLeftEye);
			imshow("Find Left Pupil", LEye);
			Mat lpgray;
			// Invert the source image and convert to grayscale
			cvtColor(~LEye, lpgray, CV_BGR2GRAY);
			imshow("Converted Left Pupil", lpgray);
			// Convert to binary image by thresholding it
			threshold(lpgray, lpgray, 220, 255, THRESH_BINARY);
			imshow("Thresh Left Pupil", lpgray);
			// Find all contours
			vector<vector<Point> > contours;
			findContours(lpgray.clone(), contours, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

			cout << "Number of contours = " << contours.size() << endl;

			// Fill holes in each contour
			drawContours(lpgray, contours, -1, CV_RGB(255,255,255), -1);
			imshow("Contours Left Pupil", lpgray);

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
				ellipse(LEye, minEllipse[indx], CV_RGB(255,0,0), 1, 8 );
				//rectangle(LEye, rect, CV_RGB(255,0,0), 2, 8 );
				imshow("Found L Pupil", LEye);
			}
		}

		if (rightEye.x > 0){
			cout << "Right Eye Rect " << dispRightEye << endl;
			//Mat REye(displayedFrame, dispRightEye);
			imshow("Find Right Pupil", REye);
			Mat rpgray;
			// Invert the source image and convert to grayscale
			cvtColor(~REye, rpgray, CV_BGR2GRAY);
			imshow("Converted Right Pupil", rpgray);
			// Convert to binary image by thresholding it
			threshold(rpgray, rpgray, 220, 255, THRESH_BINARY);
			imshow("Thresh Right Pupil", rpgray);
			// Find all contours
			vector<vector<Point> > contours2;
			findContours(rpgray.clone(), contours2, CV_RETR_EXTERNAL, CV_CHAIN_APPROX_NONE);

			cout << "Number of contours = " << contours2.size() << endl;

			// Fill holes in each contour
			drawContours(rpgray, contours2, -1, CV_RGB(255,255,255), -1);
			imshow("Contour Right Pupil", rpgray);

			// Modification to look for ellipses
			vector<RotatedRect> minRect2( contours2.size() );
			vector<RotatedRect> minEllipse2( contours2.size() );

			for( int i = 0; i < contours2.size(); i++ )
			 { minRect2[i] = minAreaRect( Mat(contours2[i]) );
				cout << "Rect2 size = " << i << ": " << contours2[i].size() << endl;
			   if( contours2[i].size() > 5 )
				 { minEllipse2[i] = fitEllipse( Mat(contours2[i]) );
					cout << "Ellipse2 size = " << i << ": " << contours2[i].size() << endl;
					cout << "Contour2 array " << Mat(contours2[i]) << endl;
				 }
			}

			// Back to Normal
			for (int i = 0; i < contours2.size(); i++)
			{
				double area = contourArea(contours2[i]);
				cout << "Contour2 area = " << i << ": " << area << endl;
				if (area > largest2){
					largest2 = area;
					indx2 = i;
				}
			}
			cout << "Largest area2 and index2 " << largest2 << ":" << indx2 << endl;
			Rect rect2 = boundingRect(contours2[indx2]);
			cout << "Rect2 = " << rect2 << endl;
			// ellipse
			if (largest2>=5){
				ellipse(REye, minEllipse2[indx2], CV_RGB(255,0,0), 1, 8 );
				//rectangle(REye, rect2, CV_RGB(255,0,0), 2, 8 );
				imshow("Found R Pupil", REye);
			}
		}

		*/

        // Show the current preprocessed face in the top-center of the display.
        int cx = (displayedFrame.cols - faceWidth) / 2;
        if (preprocessedFace.data) {
            // Get a BGR version of the face, since the output is BGR color.
            Mat srcBGR = Mat(preprocessedFace.size(), CV_8UC3);
            cvtColor(preprocessedFace, srcBGR, CV_GRAY2BGR);
            // Get the destination ROI (and make sure it is within the image!).
            //min(m_gui_faces_top + i * faceHeight, displayedFrame.rows - faceHeight);
            Rect dstRC = Rect(cx, BORDER, faceWidth, faceHeight);
            Mat dstROI = displayedFrame(dstRC);
            // Copy the pixels from src to dst.
            srcBGR.copyTo(dstROI);
        }
        // Draw an anti-aliased border around the face, even if it is not shown.
        rectangle(displayedFrame, Rect(cx-1, BORDER-1, faceWidth+2, faceHeight+2), CV_RGB(200,200,200), 1, CV_AA);

		// Show the camera frame on the screen.
        imshow(windowName, displayedFrame);

        // IMPORTANT: Wait for atleast 20 milliseconds, so that the image can be displayed on the screen!
        // Also checks if a key was pressed in the GUI window. Note that it should be a "char" to support Linux.
        char keypress = waitKey(20);  // This is needed if you want to see anything!

        if (keypress == VK_ESCAPE) {   // Escape Key
            // Quit the program!
            break;
        }

    }//end while
}


 int main(int argc, char *argv[])
{
    CascadeClassifier faceCascade;
    CascadeClassifier eyeCascade1;
    CascadeClassifier eyeCascade2;
    VideoCapture videoCapture;

    cout << "WebcamFaceEyes, by Shervin Emami (www.shervinemami.info), June 2012." << endl;
    cout << "Compiled with OpenCV version " << CV_VERSION << endl << endl;

    // Load the face and 1 or 2 eye detection XML classifiers.
    initDetectors(faceCascade, eyeCascade1, eyeCascade2);

    cout << endl;
    cout << "Hit 'Escape' in the GUI window to quit." << endl;

	 // Allow the user to specify a camera number, since not all computers will be the same camera number.
    int cameraNumber = 0;   // Change this if you want to use a different camera device.
    if (argc > 1) {
        cameraNumber = atoi(argv[1]);
    }

    // Get access to the webcam.
    initWebcam(videoCapture, cameraNumber);

    // Try to set the camera resolution. Note that this only works for some cameras on
    // some computers and only for some drivers, so don't rely on it to work!
    videoCapture.set(CV_CAP_PROP_FRAME_WIDTH, DESIRED_CAMERA_WIDTH);
    videoCapture.set(CV_CAP_PROP_FRAME_HEIGHT, DESIRED_CAMERA_HEIGHT);

    // Create a GUI window for display on the screen.
    namedWindow(windowName); // Resizable window, might not work on Windows.
    // Get OpenCV to automatically call my "onMouse()" function when the user clicks in the GUI window.
   // setMouseCallback(windowName, onMouse, 0);

    // Run Face Recogintion interactively from the webcam. This function runs until the user quits.
    recognizeUsingWebcam(videoCapture, faceCascade, eyeCascade1, eyeCascade2);

    return 0;
}

