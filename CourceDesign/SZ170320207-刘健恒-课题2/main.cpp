#include <iostream>
#include <opencv2/opencv.hpp>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>
using namespace std;
using namespace cv;
using namespace Eigen;

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
	template_data_init.pyramid_order = pyramid_order;
	return template_data_init;
}

//得到每个角度的模板信息
template_data NCC_Matching_Init(template_data template_init, Mat templateImg, int angle)      
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

	outputTemplate.firstItem = Mat(rows, cols, CV_32FC1);
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

//求图像上某一点的NCC
float NCC_Matching_once(Mat inputImage, template_data inputTemplate, int x, int y)           
{
	if (x + inputTemplate.cols >= inputImage.cols || y + inputTemplate.rows >= inputImage.rows)
	{
		return 0;
	}
	float m_f = 0;                 //图像每一模板大小的均值
	for (int u = 0; u < inputTemplate.rows; u++)
	{
		for (int v = 0; v < inputTemplate.cols; v++)
		{
			if (inputTemplate.template_bool.at<uchar>(u, v) > 0)
			{
				m_f += inputImage.at<uchar>(y + u, x + v);
			}

		}
	}
	m_f /= inputTemplate.n;

	float s_f = 0;                 //图像每一模板大小的标准差
	for (int u = 0; u < inputTemplate.rows; u++)
	{
		for (int v = 0; v < inputTemplate.cols; v++)
		{
			if (inputTemplate.template_bool.at<uchar>(u, v) > 0)
			{
				s_f += pow(inputImage.at<uchar>(y + u, x + v) - m_f, 2);
			}
		}
	}
	s_f /= inputTemplate.n;
	s_f = sqrt(s_f);
	float NCCxN = 0;

	for (int u = 0; u < inputTemplate.rows; u++)
	{
		for (int v = 0; v < inputTemplate.cols; v++)
		{
			if (inputTemplate.template_bool.at<uchar>(u, v) > 0)
			{
				NCCxN += inputTemplate.firstItem.at<float>(u, v) * (inputImage.at<uchar>(y + u, x + v) - m_f);
			}
		}
	}
	float NCC = (NCCxN / inputTemplate.n) / s_f;
	return NCC;
}

//进行NCC匹配得到NCC最大值对应点位
vector<float> NCC_Matching(Mat inputImage, template_data inputTemplate)          
{
	int image_rows = inputImage.rows;
	int image_cols = inputImage.cols;


	float NCC_max = 0; int Row_max = 0; int Col_max = 0;


	float NCC = 0;
	vector<float> NCC_data;

	for (int i = 0; i <= image_rows - inputTemplate.rows; i++)
	{
		for (int j = 0; j <= image_cols - inputTemplate.cols; j++)
		{

			NCC = NCC_Matching_once(inputImage, inputTemplate, j, i);

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

//金字塔NCC匹配
vector<float> pyramid_matching(Mat inputImage, vector<template_data> vect_template_data, float threshold_NCC = 0.7)         
{
	int definition_angle = vect_template_data.size();
	int step = vect_template_data[0].step;
	int pyramid_order = vect_template_data[0].pyramid_order;
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
			NCC_max = NCC_data_temp.back();
			NCC_max_data.clear();
			NCC_max_data.push_back(pow(2, pyramid_order - 1) * (NCC_data_temp[0]));
			NCC_max_data.push_back(pow(2, pyramid_order - 1) * (NCC_data_temp[1]));
			NCC_max_data.push_back(i * step);
			NCC_max_data.push_back(NCC_max);

		}
	}
	return NCC_max_data;
}

//金字塔初始化得到各层金字塔各个角度的模板
vector<template_data> pyramid_init(Mat templateImage, int definition_angle = 180, int step = 8, int pyramid_order = 7)         
{
	vector<template_data> vect_template_data;
	template_data template_data_init;
	template_data template_data_temp;
	template_data_init.step = step;

	int num_rotate = definition_angle / step;
	Mat rotatedImage;
	for (int i = 0; i <= num_rotate; ++i)
	{
		rotatedImage = rotate_Image(templateImage, i * step);

		for (int j = 0; j < pyramid_order - 1; ++j)
		{
			pyrDown(rotatedImage, rotatedImage, Size(rotatedImage.cols / 2, rotatedImage.rows / 2));
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

Mat rotate_Image(Mat inputImage, float angle)        //图片旋转
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

//由曲面拟合求亚像素点
vector<float> SurfaceFitting(vector<float> NCC)          
{
	Matrix<float, 9, 1> F;
	Matrix<float, 9, 6> A;
	F << NCC[0], NCC[1], NCC[2], NCC[3], NCC[4], NCC[5], NCC[6], NCC[7], NCC[8];
	A << 0, 0, 0, 0, 0, 1,
		0, 1, 0, 0, 1, 1,
		0, 4, 0, 0, 2, 1,
		1, 0, 0, 1, 0, 1,
		1, 1, 1, 1, 1, 1,
		1, 4, 2, 1, 2, 1,
		4, 0, 0, 2, 0, 1,
		4, 1, 2, 2, 1, 1,
		4, 4, 4, 2, 2, 1;
	Matrix<float, 6, 1> P;
	Matrix<float, 6, 6> X;
	X = A.transpose() * A;
	P = X.inverse() * A.transpose() * F;
	float xc, yc;
	xc = -(2 * P(1, 0) * P(3, 0) - P(2, 0) * P(4, 0)) / (4 * P(0, 0) * P(1, 0) - P(2, 0) * P(2, 0));
	yc = -(2 * P(0, 0) * P(4, 0) - P(2, 0) * P(3, 0)) / (4 * P(0, 0) * P(1, 0) - P(2, 0) * P(2, 0));
	if (abs(xc) > 3 || abs(yc) > 3)
	{
		xc = 1;
		yc = 1;
	}
	vector<float> SurfaceFitting;
	SurfaceFitting.push_back(xc);
	SurfaceFitting.push_back(yc);
	return SurfaceFitting;
}

//由曲线拟合得到更高精度的角度信息
float AngleCurveFitting(Matrix<float, 3, 2>AngleandNCC)       
{
	Matrix<float, 3, 3> D;
	Matrix<float, 3, 3> D1;
	Matrix<float, 3, 3> D2;
	Matrix<float, 3, 3> D3;
	D << pow(AngleandNCC(0, 0), 2), AngleandNCC(0, 0), 1,
		pow(AngleandNCC(1, 0), 2), AngleandNCC(1, 0), 1,
		pow(AngleandNCC(2, 0), 2), AngleandNCC(2, 0), 1;
	D1 << AngleandNCC(0, 1), AngleandNCC(0, 0), 1,
		AngleandNCC(1, 1), AngleandNCC(1, 0), 1,
		AngleandNCC(2, 1), AngleandNCC(2, 0), 1;
	D2 << pow(AngleandNCC(0, 0), 2), AngleandNCC(0, 1), 1,
		pow(AngleandNCC(1, 0), 2), AngleandNCC(1, 1), 1,
		pow(AngleandNCC(2, 0), 2), AngleandNCC(2, 1), 1;
	D3 << pow(AngleandNCC(0, 0), 2), AngleandNCC(0, 0), AngleandNCC(0, 1),
		pow(AngleandNCC(1, 0), 2), AngleandNCC(1, 0), AngleandNCC(1, 1),
		pow(AngleandNCC(2, 0), 2), AngleandNCC(2, 0), AngleandNCC(2, 1);
	float a, b, c;
	a = D1.determinant() / D.determinant();
	b = D2.determinant() / D.determinant();
	c = D3.determinant() / D.determinant();
	float angle;
	angle = -b / (2 * a);
	return angle;
}

int main(int argc, char** argv)
{
	//模板输入及初始化
	Mat templateImg = imread("../image/pattern.bmp");
	cvtColor(templateImg, templateImg, COLOR_RGB2GRAY);
	int step = 8;
	vector<template_data> vect_template_data_top = pyramid_init(templateImg, 180, step, 5);
	vector<template_data> vect_template_data_bottom = pyramid_init(templateImg, 180, 1, 1);

	for (int i = 1; i <= 34; i++)
	{
		//输入图像
		Mat input_Image = imread("../image/IMAGEB" + to_string(i) + ".bmp");
		cvtColor(input_Image, input_Image, COLOR_RGB2GRAY);
		cout << "第" << i << "张图片匹配结果:" << endl;
		//计时
		double time_consumed = static_cast<double>(cv::getTickCount());
		//由金字塔NCC匹配得到模板所在的位置及角度的估计值
		vector<float> NCC_max_data = pyramid_matching(input_Image, vect_template_data_top);
		float pos_x = NCC_max_data[0];
		float pos_y = NCC_max_data[1];
		int angle = NCC_max_data[2];
		//对底层金字塔构建5X5的ROI找到NCC最大值点
		float pos_x_max = pos_x;
		float pos_y_max = pos_y;
		float NCC_max = 0;
		float NCC_temp = 0;
		for (int i = -2; i <= 2; i++)
		{
			for (int j = -2; j <= 2; j++)
			{
				NCC_temp = NCC_Matching_once(input_Image, vect_template_data_bottom[angle], pos_x + j, pos_y + i);
				if (NCC_temp > NCC_max)
				{
					NCC_max = NCC_temp;
					pos_x_max = pos_x + j;
					pos_y_max = pos_y + i;
				}
			}
		}
		//对NCC最大值点求出其八邻域的NCC值进行曲面拟合得到亚像素坐标
		vector<float> NCC;
		for (int i = -1; i <= 1; i++)
		{
			for (int j = -1; j <= 1; j++)
			{
				NCC.push_back(NCC_Matching_once(input_Image, vect_template_data_bottom[angle], pos_x_max + j, pos_y_max + i));
			}
		}
		vector<float> fitting_pos = SurfaceFitting(NCC);
		pos_x_max = fitting_pos[0] + pos_x - 1;
		pos_y_max = fitting_pos[1] + pos_y - 1;
		//结束计时
		time_consumed = (cv::getTickCount() - time_consumed) / cv::getTickFrequency();
		cout << "耗时：" << time_consumed << " s" << endl;
		//对角度估计值取+-4邻域的NCC值得到NCC最大值点
		NCC_max = 0;
		NCC_temp = 0;
		float angle_max = angle;
		for (int i = angle - 4; i <= angle + 4; ++i)
		{
			int temp = i;
			if (i < 0)
			{
				temp += 180;
			}
			if (i > 179)
			{
				temp -= 180;
			}
			NCC_temp = NCC_Matching_once(input_Image, vect_template_data_bottom[temp], pos_x_max, pos_y_max);
			if (NCC_temp > NCC_max)
			{
				NCC_max = NCC_temp;
				angle_max = i;
			}
		}
		//取角度最大值的+-1邻域NCC值拟合曲线得到更精确角度信息
		NCC.clear();
		for (int i = angle_max - 1; i <= angle_max + 1; ++i)
		{
			int temp = i;
			if (i < 0)
			{
				temp += 180;
			}
			if (i > 179)
			{
				temp -= 180;
			}
			NCC.push_back(NCC_Matching_once(input_Image, vect_template_data_bottom[temp], pos_x_max, pos_y_max));
		}
		Matrix<float, 3, 2> angle_matrix;
		angle_matrix << angle_max - 1, NCC[0],
			            angle_max, NCC[1],
			            angle_max + 1, NCC[2];
		float fitting_angle = AngleCurveFitting(angle_matrix);
		//画图
		float radian = (float)fitting_angle / 180.0 * CV_PI;
		int angle_idx = (int)angle / step;
		float template_width = templateImg.cols;
		float template_height = templateImg.rows;
		float rotated_template_width = vect_template_data_top[angle_idx].cols * pow(2, vect_template_data_top[angle_idx].pyramid_order - 1);
		float rotated_template_height = vect_template_data_top[angle_idx].rows * pow(2, vect_template_data_top[angle_idx].pyramid_order - 1);
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
		//输出结果
		if (NCC_max == 0)
		{
			cout << "匹配失败！" << endl;
		}
		else
		{
			cout << "[x,y] = [" << pos_x_max << "," << pos_y_max << "]" << endl;
			cout << "Angle = " << fitting_angle << "°" << endl;
			cout << "NCC_max = " << NCC_max << endl;
			imshow("output" + to_string(i), input_Image);
		}
	}
	waitKey(0);
	return 0;
}