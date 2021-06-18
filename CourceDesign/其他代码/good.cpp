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

Mat rotate_Image(Mat inputImage, float angle);

struct template_data
{
	Mat template_bool;
	int height_init = 0;
	int width_init = 0;
	int n = 0;

	Mat templateImg;
	int rows = 0;
	int cols = 0;


	int angle = 0;
	int step = 8;
	int pyramid_order = 0;



	Mat firstItem;
};

template_data template_Init(Mat templateImg_Init, int pyramid_order)
{
	template_data template_data_init;
	template_data_init.height_init = templateImg_Init.rows;
	template_data_init.width_init = templateImg_Init.cols;
	template_data_init.n = templateImg_Init.rows * templateImg_Init.cols;
	template_data_init.pyramid_order = pyramid_order;                 //感觉没用到
	return template_data_init;
}

template_data NCC_Matching_Init(template_data template_init, Mat templateImg, int angle)          //template_data&没看懂
{
	template_data outputTemplate = template_init;
	outputTemplate.templateImg = templateImg;

	outputTemplate.angle = angle;
	float radian = 0;
	bool flag = 0;
	if (0 <= angle && angle < 90)
	{
		radian = (float)(angle / 180.0 * CV_PI);
		flag = 0;

	}
	else if (90 <= angle && angle <= 180)
	{
		radian = (float)((angle - 90) / 180.0 * CV_PI);
		flag = 1;
	}

	int rows = templateImg.rows;
	int cols = templateImg.cols;

	outputTemplate.rows = templateImg.rows;
	outputTemplate.cols = templateImg.cols;
	outputTemplate.template_bool = Mat::zeros(rows, cols, CV_8UC1);
	int height_init = outputTemplate.height_init;
	int width_init = outputTemplate.width_init;
	for (int u = 0; u < rows; u++)
	{
		for (int v = 0; v < cols; v++)
		{
			if (flag == 0)
			{
				if ((v + tan(radian) * u >= height_init * sin(radian)) && ((v - 1 / tan(radian) * u) <= height_init * sin(radian)) && (v + tan(radian) * u <= height_init * sin(radian) + width_init * cos(radian) + width_init * tan(radian) * sin(radian)) && ((v - 1 / tan(radian) * u) >= -1 / tan(radian) * height_init * cos(radian)))
				{
					outputTemplate.template_bool.at<uchar>(u, v) = 255;
				}
			}
			else
			{
				if ((v + tan(radian) * u >= width_init * sin(radian)) && ((v - 1 / tan(radian) * u) <= width_init * sin(radian)) && (v + tan(radian) * u <= width_init * sin(radian) + height_init * cos(radian) + height_init * tan(radian) * sin(radian)) && ((v - 1 / tan(radian) * u) >= -1 / tan(radian) * width_init * cos(radian)))
				{
					outputTemplate.template_bool.at<uchar>(u, v) = 255;
				}
			}

		}
	}


	float m_t = 0;                //模板灰度均值
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (outputTemplate.template_bool.at<uchar>(i, j) > 0)
			{
				m_t += templateImg.at<uchar>(i, j);
			}
		}
	}
	m_t /= outputTemplate.n;

	float s_t = 0;                 //模板灰度标准差
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (outputTemplate.template_bool.at<uchar>(i, j) > 0)
			{
				s_t += pow(templateImg.at<uchar>(i, j) - m_t, 2);
			}
		}
	}
	s_t /= outputTemplate.n;
	s_t = sqrt(s_t);

	outputTemplate.firstItem = Mat(rows, cols, CV_32FC1);         //CV_32FC1是啥
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			if (outputTemplate.template_bool.at<uchar>(i, j) > 0)
			{
				outputTemplate.firstItem.at<float>(i, j) = (templateImg.at<uchar>(i, j) - m_t) / s_t;
			}
		}
	}
	return outputTemplate;
}

vector<float> NCC_Matching(Mat inputImage, template_data inputTemplate, float threshold_NCC = 0.7)       //为啥这不用&
{
	int image_rows = inputImage.rows;
	int image_cols = inputImage.cols;

	float NCCxN = 0;                     //不懂
	float NCC = 0;
	bool isFind = false;                  //用来干嘛的
	float NCC_max = 0; int Row_max = 0; int Col_max = 0;

	float m_f = 0;
	float s_f = 0;

	vector<float> NCC_data;

	for (int i = 0; i <= image_rows - inputTemplate.rows; i++)
	{
		for (int j = 0; j <= image_cols - inputTemplate.cols; j++)
		{

			Mat template_bool = Mat::zeros(inputTemplate.rows, inputTemplate.cols, CV_8UC1);

			m_f = 0;                 //图像每一模板大小的均值
			for (int u = 0; u < inputTemplate.rows; u++)
			{
				for (int v = 0; v < inputTemplate.cols; v++)
				{
					if (inputTemplate.template_bool.at<uchar>(u, v) > 0)
					{
						m_f += inputImage.at<uchar>(i + u, j + v);                    //行是x嘛？
						template_bool.at<uchar>(u, v) = 255;
					}

				}
			}
			//imshow("1", template_bool);
			m_f /= inputTemplate.n;

			s_f = 0;                 //图像每一模板大小的标准差
			for (int u = 0; u < inputTemplate.rows; u++)
			{
				for (int v = 0; v < inputTemplate.cols; v++)
				{
					if (inputTemplate.template_bool.at<uchar>(u, v) > 0)
					{
						s_f += pow(inputImage.at<uchar>(i + u, j + v) - m_f, 2);
					}
				}
			}
			s_f /= inputTemplate.n;
			s_f = sqrt(s_f);

			NCCxN = 0;
			for (int u = 0; u < inputTemplate.rows; u++)
			{
				for (int v = 0; v < inputTemplate.cols; v++)
				{
					if (inputTemplate.template_bool.at<uchar>(u, v) > 0)
					{
						NCCxN += inputTemplate.firstItem.at<float>(u, v) * (inputImage.at<uchar>(i + u, j + v) - m_f);
					}
				}
			}
			NCC = (NCCxN / inputTemplate.n) / s_f;        //两个除号是先除前面的嘛

			if (NCC > NCC_max)
			{
				NCC_max = NCC;
				Row_max = i;
				Col_max = j;
			}
		}
	}
	
	NCC_data.push_back(Col_max);
	NCC_data.push_back(Row_max);
	NCC_data.push_back(NCC_max);

	return NCC_data;
}


vector<float> pyramid_matching(Mat inputImage, vector<template_data> vect_template_data, float threshold_NCC = 0.7)
{
	int definition_angle = vect_template_data.size();                       //不知道是啥
	int step = vect_template_data[0].step;
	int pyramid_order = vect_template_data[0].pyramid_order;                    //这个【0】是啥意思  order是几层的意思吗
	float NCC_max = 0;
	vector<float> NCC_max_data;

	for (int j = 0; j < pyramid_order - 1; ++j)
	{
		pyrDown(inputImage, inputImage, Size(inputImage.cols / 2, inputImage.rows / 2));          //先对图像进行高斯平滑，然后再进行降采样
	}
	for (int i = 0; i < definition_angle; ++i)
	{
		vector<float> NCC_data_temp = NCC_Matching(inputImage, vect_template_data[i]);

		if (NCC_data_temp.back() > NCC_max)
		{
			NCC_max = NCC_data_temp.back();                     //back是最后一位吗 
			NCC_max_data.clear();
			NCC_max_data.push_back(pow(2, pyramid_order - 1) * (NCC_data_temp[0]));        //不懂 为什么是2的这么多次方乘这个是什么
			NCC_max_data.push_back(pow(2, pyramid_order - 1) * (NCC_data_temp[1]));
			NCC_max_data.push_back(i* step);
			NCC_max_data.push_back(NCC_max);

			cout << "[x,y] = " << NCC_max_data[0] << "," << NCC_max_data[1] << endl;
			cout << "Angle = " << NCC_max_data[2] << endl;
			cout << "NCC_max = " << NCC_max_data[3] << endl;
		}
	}

	return NCC_max_data;

}

vector<template_data> pyramid_init(Mat templateImage, int definition_angle = 180, int step = 8, int pyramid_order = 7)                 //没太懂 和上面那个角度的有啥区别
{
	vector<template_data> vect_template_data;
	template_data template_data_init;
	template_data template_data_temp;
	template_data_init.step = step;

	int num_rotate = definition_angle / step;
	Mat rotatedImage;
	for (int i = 0; i <= num_rotate; ++i)                   //++i是啥
	{
		rotatedImage = rotate_Image(templateImage, i * step);

		for (int j = 0; j < pyramid_order - 1; ++j)
		{
			pyrDown(rotatedImage, rotatedImage, Size(rotatedImage.cols / 2, rotatedImage.rows / 2));               //感觉这里没用到循环j
		}
		if (i == 0)
		{
			template_data_init = template_Init(rotatedImage, pyramid_order);
		}
		template_data_temp = NCC_Matching_Init(template_data_init, rotatedImage, i * step);
		vect_template_data.push_back(template_data_temp);
	}

	return vect_template_data;
}

Mat rotate_Image(Mat inputImage, float angle)
{
	float radian = (float)(angle / 180.0 * CV_PI);

	//填充图像
	int maxBorder = (int)(max(inputImage.cols, inputImage.rows) * 1.414); //即为sqrt(2)*max
	int dx = (maxBorder - inputImage.cols) / 2;
	int dy = (maxBorder - inputImage.rows) / 2;
	Mat outputImage;
	copyMakeBorder(inputImage, outputImage, dy, dy, dx, dx, BORDER_CONSTANT);

	//旋转
	Point2f center((float)(outputImage.cols / 2), (float)(outputImage.rows / 2));
	Mat affine_matrix = getRotationMatrix2D(center, angle, 1.0);//求得旋转矩阵
	warpAffine(outputImage, outputImage, affine_matrix, outputImage.size());

	//计算图像旋转之后包含图像的最大的矩形
	float sinVal = abs(sin(radian));
	float cosVal = abs(cos(radian));
	Size targetSize((int)(inputImage.cols * cosVal + inputImage.rows * sinVal),
		(int)(inputImage.cols * sinVal + inputImage.rows * cosVal));

	//剪掉多余边框
	int x = (outputImage.cols - targetSize.width) / 2;
	int y = (outputImage.rows - targetSize.height) / 2;
	Rect rect(x, y, targetSize.width, targetSize.height);
	outputImage = outputImage(rect);
	return outputImage;
}

int main(int argc, char** argv)
{
	Mat templateImg = imread("../images/pattern.bmp");
	cvtColor(templateImg, templateImg, COLOR_RGB2GRAY);                                     //这个不懂 取templateImg的图像得到他的灰度值图像命名为templateImg吗

	Mat input_Image = imread("../images/1.bmp");
	cvtColor(input_Image, input_Image, COLOR_RGB2GRAY);

	int step = 8;

	vector<template_data> vect_template_data_top = pyramid_init(templateImg, 180, step, 5);             //不懂
	vector<template_data> vect_template_data_button = pyramid_init(templateImg, 180, 1, 1);

	//计时
	double time_consumed = static_cast<double>(cv::getTickCount());            //后面这不懂

	vector<float> NCC_max_data = pyramid_matching(input_Image, vect_template_data_top);    //不懂      
	time_consumed = (cv::getTickCount() - time_consumed) / cv::getTickFrequency();
	cout << "耗时：" << time_consumed << " s" << endl;

	float pos_x = NCC_max_data[0];
	float pos_y = NCC_max_data[1];
	int angle = NCC_max_data[2]/ step;
	float radian = (float)angle / 180.0 * CV_PI;
	float template_width = templateImg.cols;
	float template_height = templateImg.rows;
	float rotated_template_width = vect_template_data_top[angle].cols * pow(2, vect_template_data_top[angle].pyramid_order - 1);
	float rotated_template_height = vect_template_data_top[angle].rows * pow(2, vect_template_data_top[angle].pyramid_order - 1);

	float radian_template = atan2f(rotated_template_height, template_width);
	float length = sqrt(template_width * template_width + template_height * template_height) / 2;

	int centerCol = pos_x + rotated_template_width / 2, centerRow = pos_y + rotated_template_width / 2;
	float theta0 = atan2f(template_height, template_width);
	float phi1 = theta0 + radian, phi2 = theta0 - radian;

	line(input_Image, Point(centerCol - length * cosf(phi1), centerRow + length * sinf(phi1)), Point(centerCol + length * cosf(phi2), centerRow + length * sinf(phi2)), Scalar(255, 0, 0), 2);
	line(input_Image, Point(centerCol + length * cosf(phi2), centerRow + length * sinf(phi2)), Point(centerCol + length * cosf(phi1), centerRow - length * sinf(phi1)), Scalar(255, 0, 0), 2);
	line(input_Image, Point(centerCol + length * cosf(phi1), centerRow - length * sinf(phi1)), Point(centerCol - length * cosf(phi2), centerRow - length * sinf(phi2)), Scalar(255, 0, 0), 2);
	line(input_Image, Point(centerCol - length * cosf(phi2), centerRow - length * sinf(phi2)), Point(centerCol - length * cosf(phi1), centerRow + length * sinf(phi1)), Scalar(255, 0, 0), 2);

	circle(input_Image, Point(centerCol, centerRow), 3, Scalar(255, 0, 0), -1);


	cout << "[x,y] = " << NCC_max_data[0] << " , " << NCC_max_data[1] << endl;
	cout << "Angle = " << NCC_max_data[2] << endl;
	cout << "NCC_max = " << NCC_max_data[3] << endl;
	imshow("output", input_Image);
	waitKey(0);

	return 0;
}