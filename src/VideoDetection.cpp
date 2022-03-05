#include <opencv2/imgcodecs.hpp>
#include <opencv2/highgui.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/objdetect.hpp>
#include <iostream>
#include <direct.h>

using namespace cv;

void detect(Mat frame);
void threadDetect(CascadeClassifier cf, Mat frame, std::vector<Rect>& detections, double scaleFactor, int minNeighbors);

CascadeClassifier faceCF;
CascadeClassifier catCF;

int main(int argc, const char** argv) {

	//Setup
	VideoCapture cap(0);
	Mat frame;
	if (!cap.isOpened()) {
		std::cout << "Could not connect to the camera." << std::endl;
	}

	if (!faceCF.load("resources/haarcascade_frontalface_default.xml")) {
		std::cout << "Face Classifier XML file could not be loaded." << std::endl;
		return -1;
	}

	if (!catCF.load("resources/haarcascade_frontalcatface.xml")) {
		std::cout << "Cat Classifier XML file could not be loaded." << std::endl;
		return -1;
	}

	struct stat dirInfo;
	if (stat("detections", &dirInfo) != 0) {
		if (_mkdir("detections") != 0) {
			std::cout << "Could not create directory for detections." << std::endl;
			return -1;
		}
	}
	
	//Start
	while (cap.read(frame)) {

		if (frame.empty()) {
			std::cout << "No frame captured -> Break!" << std::endl;
			break;
		}

		detect(frame);
		imshow("Detection capture", frame);

		// Break with 'esc'
		if (waitKey(1) == 27)
		{
			break;
		}
	}
	return 0;
}

void detect(Mat frame) {
	// preprocessing for better recognitions
	Mat frame_gray;
	cvtColor(frame, frame_gray, COLOR_BGR2GRAY);
	equalizeHist(frame_gray, frame_gray);

	std::vector<Rect> faces;
	std::vector<Rect> cats;
	faceCF.detectMultiScale(frame_gray, faces, 1.1, 10);
	catCF.detectMultiScale(frame_gray, cats, 1.1, 10);
	//std::thread fT(threadDetect, faceCF, frame_gray, std::ref(faces), 1.1, 10);
	//std::thread cT(threadDetect, catCF, frame_gray, std::ref(cats), 1.1, 10);
	
	//fT.join();
	for (size_t i = 0; i < faces.size(); i++) {
		Mat imgCrop = frame(faces[i]);
		imwrite("detections/face_" + std::to_string(i) + ".png", imgCrop);
		rectangle(frame, faces[i].tl(), faces[i].br(), Scalar(255, 0, 255), 3);
		putText(frame, "Face", { faces[i].tl().x, faces[i].tl().y-5 }, QT_FONT_NORMAL, 1.25, Scalar(255, 0, 255), 2);
	}

	//cT.join();
	for (size_t i = 0; i < cats.size(); i++) {
		Mat imgCrop = frame(cats[i]);
		imwrite("detections/cat_" + std::to_string(i) + ".png", imgCrop);
		rectangle(frame, cats[i].tl(), cats[i].br(), Scalar(255, 0, 0), 3);
		putText(frame, "Cat", { cats[i].tl().x, cats[i].tl().y-5 }, QT_FONT_NORMAL, 1.25, Scalar(255, 0, 0), 2);
	}
}

void threadDetect(CascadeClassifier cf, Mat frame, std::vector<Rect> &detections, double scaleFactor, int minNeighbors) {
	cf.detectMultiScale(frame, detections, scaleFactor, minNeighbors);
}