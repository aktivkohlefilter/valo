#pragma once
#ifndef CAPTURE_H
#define CAPTURE_H

#include <Windows.h>
#include <stdio.h>
#include <vector>
#include <opencv2/core/mat.hpp>
#include <opencv2/imgproc.hpp>

// Takes a screenshot and saves it inside a Screenshot class lal //
class Capture
{
private:
	HWND hWindow; //Handle of window
	char* windowName; //Name of window
	uchar* BGR;
	HDC hScreen, g_stcMemDc;
	BITMAPINFOHEADER bmi = { 0 };
	void release(HWND& hwnd, HDC& hdc, HDC& captureDC, HBITMAP& hBmp);



public:
	Capture(char* windowName);
	~Capture();

	bool isWindowRunning();
	cv::Mat hwnd2mat();
	void InitDx();
	bool screenshotGDI(cv::Mat& outMat); //Obtains screenshot via win32. Outputs data into _LastFrame struct
	void switchToWindow();
	void initScreenshot();
};


#endif 
