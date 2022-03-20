#ifdef _WIN32
#define OPENCV
#define GPU
#endif
#include <iostream>
#include <opencv2/opencv.hpp>
#include <windows.h>
#include <fstream>
#include<time.h>
#include <Windows.h>
#include <string>
#include <codecvt>
#include"arduino.h"
#include"Capture.h"
#define PROCESS_NAME L"VALORANT  " 
namespace Settings
{
	float CONFIDENCE_THRESHOLD = 0.4;
	float NMS_THRESHOLD = 0.3;
	int NUM_CLASSES = 1;
	float speed = 1;
	float flickSpeed = 1000;
	float flickDistanceFactor = 2;

	float horizontal_offset = 0.5;          // Fractional offset(between 0 and 1) from left side of the detection.
	float vertical_offset = 0.1;            // Fractional offset(between 0 and 1) from top top side of the detection.
	int screen_dimensions[] = { GetSystemMetrics(SM_CXSCREEN), GetSystemMetrics(SM_CYSCREEN) };
	int input_dimensions[] = { 600, 200 };
	bool degub = false;
	//Mode mode = Mode::Track;
	int aimKey = VK_MBUTTON;
}
Arduino ard;
const cv::Scalar colors[] = {
	{0, 255, 255},
	{255, 255, 0},
	{0, 255, 0},
	{255, 0, 0}
};
static std::wstring s2ws(const std::string& str)
{
	typedef std::codecvt_utf8<wchar_t> convert_typeX;
	std::wstring_convert<convert_typeX, wchar_t> converterX;

	return converterX.from_bytes(str);
}
using namespace std;
using namespace cv;

uchar* BGR;
HDC hScreen, g_stcMemDc;
int Screenshot_W = 600;
int Screenshot_H = 200;

BITMAPINFOHEADER bmi = { 0 };

//constexpr const char* image_path = "E:/3.jpg";//´ý¼ì²âÍ¼Æ¬
constexpr const char* darknet_cfg = "custom-yolov4-tiny-detector(2).cfg";//ÍøÂçÎÄ¼þ
constexpr const char* darknet_weights = "custom-yolov4-tiny-detector_best(2).weights";//ÑµÁ·Ä£ÐÍ
constexpr const char* darknet_names = "classes2.txt"; //
std::vector<std::string> class_labels;//Àà±êÇ©

// Initialize the parameters
float confThreshold = 0.25; // Confidence threshold
float nmsThreshold = 0.4;  // Non-maximum suppression threshold
int inpWidth = 416;        // Width of network's input image
int inpHeight = 416;       // Height of network's input image


// Get the names of the output layers
vector<String> getOutputsNames(const dnn::Net& net)
{
	static vector<String> names;
	if (names.empty())
	{
		//Get the indices of the output layers, i.e. the layers with unconnected outputs
		vector<int> outLayers = net.getUnconnectedOutLayers();

		//get the names of all the layers in the network
		vector<String> layersNames = net.getLayerNames();

		// Get the names of the output layers in names
		names.resize(outLayers.size());
		for (size_t i = 0; i < outLayers.size(); ++i)
			names[i] = layersNames[outLayers[i] - 1];
	}
	return names;
}

static std::mutex s_Frame;
void Screenshot(cv::Mat* outMat) {
	Capture screen((char*)"TankWindowClass");
	while (!screen.isWindowRunning())
	{
		std::cout << ".";
		Sleep(1000);
	}
	screen.InitDx();
	screen.initScreenshot();
	cv::Mat frame;
	while (true)
	{
		frame = screen.hwnd2mat().clone();
		std::lock_guard<std::mutex> lock(s_Frame);
		cv::cvtColor(frame, *outMat, cv::ColorConversionCodes::COLOR_BGRA2BGR, 3);
	}
}

// Draw the predicted bounding box
void drawPred(int classId, float conf, int left, int top, int right, int bottom, Mat& frame)
{
	//Draw a rectangle displaying the bounding box
	rectangle(frame, Point(left, top), Point(right, bottom), Scalar(0, 0, 255));

	//Get the label for the class name and its confidence
	string label = format("%.2f", conf);
	if (!class_labels.empty())
	{
		//assert(classId < (int)classes.size());
		label = class_labels[classId] + ":" + label;
	}

	//Display the label at the top of the bounding box
	int baseLine;
	Size labelSize = getTextSize(label, FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
	top = max(top, labelSize.height);
	putText(frame, label, Point(left, top), FONT_HERSHEY_SIMPLEX, 0.5, Scalar(255, 255, 255));

}


// Remove the bounding boxes with low confidence using non-maxima suppression
void postprocess(Mat& frame, const vector<Mat>& outs)
{
	vector<int> classIds;
	vector<float> confidences;
	vector<Rect> boxes;

	for (size_t i = 0; i < outs.size(); ++i)
	{
		// Scan through all the bounding boxes output from the network and keep only the
		// ones with high confidence scores. Assign the box's class label as the class
		// with the highest score for the box.
		float* data = (float*)outs[i].data;
		for (int j = 0; j < outs[i].rows; ++j, data += outs[i].cols)
		{
			Mat scores = outs[i].row(j).colRange(5, outs[i].cols);
			Point classIdPoint;
			double confidence;
			// Get the value and location of the maximum score
			minMaxLoc(scores, 0, &confidence, 0, &classIdPoint);
			if (confidence > confThreshold)
			{
				int centerX = (int)(data[0] * frame.cols);
				int centerY = (int)(data[1] * frame.rows);
				int width = (int)(data[2] * frame.cols);
				int height = (int)(data[3] * frame.rows);
				int left = centerX - width / 2;
				int top = centerY - height / 2;

				classIds.push_back(classIdPoint.x);
				confidences.push_back((float)confidence);
				boxes.push_back(Rect(left, top, width, height));
			}
		}
	}

	// Perform non maximum suppression to eliminate redundant overlapping boxes with
	// lower confidences
	vector<int> indices;
	dnn::NMSBoxes(boxes, confidences, confThreshold, nmsThreshold, indices);
	for (size_t i = 0; i < indices.size(); ++i)
	{
		int idx = indices[i];
		Rect box = boxes[idx];
		drawPred(classIds[idx], confidences[idx], box.x, box.y,
			box.x + box.width, box.y + box.height, frame);
	}
}
static std::mutex s_Vector;
void GetDetections(cv::Mat* output, cv::Mat* outMat, std::vector<float>* scores, std::vector<cv::Rect>* boxes) {
	const auto num_boxes = output->rows;
	for (int i = 0; i < num_boxes; i++)
	{
		auto x = output->at<float>(i, 0) * outMat->cols;
		auto y = output->at<float>(i, 1) * outMat->rows;
		auto width = output->at<float>(i, 2) * outMat->cols;
		auto height = output->at<float>(i, 3) * outMat->rows;
		cv::Rect rect(x - width / 2, y - height / 2, width, height);

		auto confidence = *output->ptr<float>(i, 5);
		if (confidence >= 0.4)
		{
			std::lock_guard<std::mutex> lock(s_Vector);
			boxes->push_back(rect);
			scores->push_back(confidence);
		}
	}
}

int main()
{
	ard.Init(s2ws("COM3"));
	// ¼ÓÔØÄ£ÐÍ
	cv::dnn::Net net = cv::dnn::readNetFromDarknet(darknet_cfg, darknet_weights);
	net.setPreferableBackend(dnn::DNN_BACKEND_CUDA);
	net.setPreferableTarget(dnn::DNN_TARGET_CUDA);
	//net.setPreferableTarget(dnn::DNN_TARGET_CPU);

	// ¼ÓÔØ±êÇ©¼¯
	//std::vector<std::string> classLabels;
	ifstream classNamesFile(darknet_names);
	if (classNamesFile.is_open())
	{
		string className = "";
		while (std::getline(classNamesFile, className))
			class_labels.push_back(className);
	}

	// ¶ÁÈ¡´ý¼ì²âÍ¼Æ¬
	//cv::Mat img = cv::imread(image_path);
	Mat ScreenshotMat(Screenshot_H, Screenshot_W, CV_8UC3, Scalar(0, 0, 255));
	auto cap_start = std::chrono::steady_clock::now();
	std::thread captureScreen(Screenshot, &ScreenshotMat);
	auto cap_end = std::chrono::steady_clock::now();
	float speedMultiplier = (6.4 - 10) / (10);
	while (cv::waitKey(1) < 1) {
		auto total_start = std::chrono::steady_clock::now();
		cv::Mat blob = cv::dnn::blobFromImage(ScreenshotMat, 1.0 / 255.0, { inpWidth, inpHeight }, 0.00392, true);
		net.setInput(blob);

		// ¼ì²â
		vector<Mat> detectionMat;
		auto dnn_start = std::chrono::steady_clock::now();
		net.forward(detectionMat, getOutputsNames(net));// 6 845 1 W x H x C
		auto dnn_end = std::chrono::steady_clock::now();
		std::vector<int> indices;
		std::vector<cv::Rect> boxes;
		std::vector<float> scores;
		for (int i = 0; i < detectionMat.size(); i++)
		{
			GetDetections(&detectionMat[i], &ScreenshotMat, &scores, &boxes);
		}
		// Remove the bounding boxes with low confidence
		postprocess(ScreenshotMat, detectionMat);
		cv::dnn::NMSBoxes(boxes, scores, 0.0, 0.3, indices);
		int min_idx = -1;
		int min_distance = 0.0;
		int target_dx = 0.0;
		int target_dy = 0.0;
		int trigger = 0;
		for (size_t i = 0; i < indices.size(); ++i)
		{
			const auto& rect = boxes[indices[i]];

			int dx = (rect.x + 0.5 * rect.width - Screenshot_W / 2);
			int dy = (rect.y + 0.1 * rect.height - Screenshot_H / 2);
			//cout << rect.x << ":" << rect.y << ":" << dx << ":" << dy << ":" << rect.width<<":"<<rect.height<<endl;
			int distance = sqrt(dx * dx + dy * dy);
			if (min_idx < 0 || distance < min_distance) {
				if (distance < rect.width * (0.1 + 0.05))
					trigger = 1;
				min_idx = i;
				min_distance = distance;
				target_dx = dx;
				target_dy = dy;
			}
			if (false) {

				const auto color = colors[1];

				auto idx = indices[i];
				const auto& rect = boxes[idx];
				cv::rectangle(ScreenshotMat, cv::Point(rect.x, rect.y), cv::Point(rect.x + rect.width, rect.y + rect.height), color, 3);

				std::ostringstream label_ss;
				label_ss << class_labels[0] << ": " << std::fixed << std::setprecision(2) << scores[idx];
				auto label = label_ss.str();

				int baseline;
				auto label_bg_sz = cv::getTextSize(label.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
				cv::rectangle(ScreenshotMat, cv::Point(rect.x, rect.y - label_bg_sz.height - baseline - 10), cv::Point(rect.x + label_bg_sz.width, rect.y), color, cv::FILLED);
				cv::putText(ScreenshotMat, label.c_str(), cv::Point(rect.x, rect.y - baseline - 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(0, 0, 0));
			}
		}
		if (true) {
			if (!GetAsyncKeyState(86)) {//!GetAsyncKeyState(86)
				//int x;
				//int y;
				//smooth(roundto127(target_dx), roundto127(target_dy), &x, &y, speedMultiplier);
				if (target_dx <= 0 && target_dy <= 0)
					ard.mouseEvent(abs(target_dx), abs(target_dy), trigger, 1, 1);
				else if (target_dx > 0 && target_dy <= 0)
					ard.mouseEvent(abs(target_dx), abs(target_dy), trigger, 0, 1);
				else if (target_dx > 0 && target_dy > 0)
					ard.mouseEvent(abs(target_dx), abs(target_dy), trigger, 0, 0);
				else if (target_dx <= 0 && target_dy > 0)
					ard.mouseEvent(abs(target_dx), abs(target_dy), trigger, 1, 0);
			}
		}
		if (false)
		{
			auto total_end = std::chrono::steady_clock::now();
			float inference_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(dnn_end - dnn_start).count();
			float cap_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(cap_end - cap_start).count();
			float total_fps = 1000.0 / std::chrono::duration_cast<std::chrono::milliseconds>(total_end - total_start).count();
			cout << "I£º " << (int)inference_fps << "T£º " << (int)total_fps << "C£º " << (int)cap_fps << endl;
			std::ostringstream stats_ss;
			stats_ss << std::fixed << std::setprecision(2);
			stats_ss << "I" << inference_fps << "T" << total_fps << "C" << cap_fps;
			auto stats = stats_ss.str();

			int baseline;
			auto stats_bg_sz = cv::getTextSize(stats.c_str(), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, 1, &baseline);
			cv::rectangle(ScreenshotMat, cv::Point(0, 0), cv::Point(stats_bg_sz.width, stats_bg_sz.height + 10), cv::Scalar(0, 0, 0), cv::FILLED);
			cv::putText(ScreenshotMat, stats.c_str(), cv::Point(0, stats_bg_sz.height + 5), cv::FONT_HERSHEY_COMPLEX_SMALL, 1, cv::Scalar(255, 255, 255));

			cv::namedWindow("output");
			cv::imshow("output", ScreenshotMat);
		}
	}
}
