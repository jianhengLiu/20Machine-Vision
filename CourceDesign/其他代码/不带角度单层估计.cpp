//
// Created by chrisliu on 2020/4/9.
//

#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>

using namespace std;
using namespace cv;

struct template_data
{
	int rows = 0;
	int cols = 0;
	int n = 0;

	Mat firstItem;
};

void NCC_Matching_Init(template_data& inputTemplate, Mat templateImg)
{
	int rows = templateImg.rows;
	int cols = templateImg.cols;

	inputTemplate.rows = rows;
	inputTemplate.cols = cols;
	inputTemplate.n = rows * cols;

	float m_t = 0;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			m_t += templateImg.at<uchar>(i, j);
		}
	}
	m_t /= inputTemplate.n;

	float s_t = 0;
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			s_t += pow(templateImg.at<uchar>(i, j) - m_t, 2);
		}
	}
	s_t /= inputTemplate.n;
	s_t = sqrt(s_t);

	inputTemplate.firstItem = Mat(rows, cols, CV_32FC1);
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			inputTemplate.firstItem.at<float>(i, j) = (templateImg.at<uchar>(i, j) - m_t) / s_t;
		}
	}
}

Rect NCC_Matching(Mat inputImage, template_data inputTemplate, float threshold_NCC = 0.7)
{
	Rect foundedRect;

	int image_rows = inputImage.rows;
	int image_cols = inputImage.cols;

	float NCCxN = 0;
	float NCC = 0;
	string text = "NCC=";
	bool isFind = false;
	float NCC_max = 0; int Row_max = 0; int Col_max = 0;

	float m_f = 0;
	float s_f = 0;

	for (int i = 0; i <= image_rows - inputTemplate.rows; i++)
	{
		for (int j = 0; j <= image_cols - inputTemplate.cols; j++)
		{
			m_f = 0;
			for (int u = 0; u < inputTemplate.rows; u++)
			{
				for (int v = 0; v < inputTemplate.cols; v++)
				{
					m_f += inputImage.at<uchar>(i + u, j + v);
				}
			}
			m_f /= inputTemplate.n;

			s_f = 0;
			for (int u = 0; u < inputTemplate.rows; u++)
			{
				for (int v = 0; v < inputTemplate.cols; v++)
				{
					s_f += pow(inputImage.at<uchar>(i + u, j + v) - m_f, 2);
				}
			}
			s_f /= inputTemplate.n;
			s_f = sqrt(s_f);

			NCCxN = 0;
			for (int u = 0; u < inputTemplate.rows; u++)
			{
				for (int v = 0; v < inputTemplate.cols; v++)
				{
					NCCxN += inputTemplate.firstItem.at<float>(u, v) * (inputImage.at<uchar>(i + u, j + v) - m_f);
				}
			}
			NCC = NCCxN / inputTemplate.n / s_f;


			if (NCC > NCC_max)
			{
				NCC_max = NCC;
				Row_max = i;
				Col_max = j;
			}
		}
	}

	foundedRect = Rect(Col_max, Row_max, inputTemplate.cols, inputTemplate.rows);
	//cout << "Position=(" + to_string(Row_max) + "," + to_string(Col_max) + ")" << endl;
	//cout << "NCCmax=" << NCC_max << endl;
	return foundedRect;
}


Rect pyramid_matching(Mat inputImage, vector<template_data> vect_template_data, float threshold_NCC = 0.7)
{
	int pyramid_order = vect_template_data.size();
	for (int i = 0; i < pyramid_order - 1; ++i)
	{
		pyrDown(inputImage, inputImage, Size(inputImage.cols / 2, inputImage.rows / 2));
	}
	
	Rect foundedRect_temp = NCC_Matching(inputImage, vect_template_data.back());
	

	foundedRect_temp.x = pow(2, pyramid_order-1) * (foundedRect_temp.x);
	foundedRect_temp.y = pow(2, pyramid_order-1) * (foundedRect_temp.y);
	
	return foundedRect_temp;

}

vector<template_data> pyramid_init(Mat templateImage, int pyramid_order = 7)
{
	vector<template_data> vect_template_data;
	template_data template_data_temp;
	NCC_Matching_Init(template_data_temp, templateImage);
	vect_template_data.push_back(template_data_temp);

	for (int i = 0; i < pyramid_order-1; ++i)
	{
		pyrDown(templateImage, templateImage, Size(templateImage.cols / 2, templateImage.rows / 2));
		NCC_Matching_Init(template_data_temp, templateImage);
		vect_template_data.push_back(template_data_temp);
	}
	return vect_template_data;
}

int main(int argc, char** argv)
{
	Mat templateImg = imread("../images/pattern.bmp");
	cvtColor(templateImg, templateImg, COLOR_RGB2GRAY);
	vector<template_data> vect_template_data = pyramid_init(templateImg, 5);

	Mat input_Image = imread("../images/1.bmp");
	cvtColor(input_Image, input_Image, COLOR_RGB2GRAY);

	//计时
	double time_consumed = static_cast<double>(cv::getTickCount());

	Rect foundedRect = pyramid_matching(input_Image, vect_template_data);
	foundedRect.width = templateImg.cols;
	foundedRect.height = templateImg.rows;
	time_consumed = (cv::getTickCount() - time_consumed) / cv::getTickFrequency();
	cout << "耗时：" << time_consumed << " s" << endl;
	cout << foundedRect << endl;
	rectangle(input_Image, foundedRect, Scalar(255));
	imshow("output", input_Image);
	waitKey(0);

	return 0;
}