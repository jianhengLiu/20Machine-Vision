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
	Mat templateImg;
	int rows = 0;
	int cols = 0;
	int n = 0;

	int pyramid_order = 0;

	Mat firstItem;           
};

void NCC_Matching_Init(template_data& inputTemplate, Mat templateImg, int pyramid_order)          //template_data&û����
{
	inputTemplate.templateImg = templateImg;                           
	inputTemplate.pyramid_order = pyramid_order;                 //�о�û�õ�
	int rows = templateImg.rows;
	int cols = templateImg.cols;

	inputTemplate.rows = rows;
	inputTemplate.cols = cols;
	inputTemplate.n = rows * cols;

	float m_t = 0;                //ģ��ҶȾ�ֵ
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			m_t += templateImg.at<uchar>(i, j);
		}
	}
	m_t /= inputTemplate.n;           

	float s_t = 0;                 //ģ��Ҷȱ�׼��
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			s_t += pow(templateImg.at<uchar>(i, j) - m_t, 2);
		}
	}
	s_t /= inputTemplate.n;
	s_t = sqrt(s_t);

	inputTemplate.firstItem = Mat(rows, cols, CV_32FC1);         //CV_32FC1��ɶ
	for (int i = 0; i < rows; i++)
	{
		for (int j = 0; j < cols; j++)
		{
			inputTemplate.firstItem.at<float>(i, j) = (templateImg.at<uchar>(i, j) - m_t) / s_t;           
		}
	}
}

vector<float> NCC_Matching(Mat inputImage, template_data inputTemplate, float threshold_NCC = 0.7)       //Ϊɶ�ⲻ��&
{
	int image_rows = inputImage.rows;
	int image_cols = inputImage.cols;

	float NCCxN = 0;                     //����
	float NCC = 0;
	bool isFind = false;                  //���������
	float NCC_max = 0; int Row_max = 0; int Col_max = 0;

	float m_f = 0;
	float s_f = 0;

	for (int i = 0; i <= image_rows - inputTemplate.rows; i++)           
	{
		for (int j = 0; j <= image_cols - inputTemplate.cols; j++)
		{
			m_f = 0;                 //ͼ��ÿһģ���С�ľ�ֵ
			for (int u = 0; u < inputTemplate.rows; u++)
			{
				for (int v = 0; v < inputTemplate.cols; v++)
				{
					m_f += inputImage.at<uchar>(i + u, j + v);                    //����x�
				}
			}
			m_f /= inputTemplate.n;

			s_f = 0;                 //ͼ��ÿһģ���С�ı�׼��
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
			NCC = NCCxN / inputTemplate.n / s_f;        //�����������ȳ�ǰ�����


			if (NCC > NCC_max)
			{
				NCC_max = NCC;
				Row_max = i;
				Col_max = j;
			}
		}
	}

	vector<float> NCC_data;                  //��vector��һ����col�ڶ�����row����˼�𣿲���Ӧ��������ֵ�õ��ü�������Ϊʲôֻ��һ����
	NCC_data.push_back(Col_max);
	NCC_data.push_back(Row_max);
	NCC_data.push_back(NCC_max);

	return NCC_data;
}


vector<float> pyramid_matching(Mat inputImage, vector<template_data> vect_template_data, float threshold_NCC = 0.7)
{
	int definition_angle = vect_template_data.size();                       //��֪����ɶ
	int pyramid_order = vect_template_data[0].pyramid_order;                    //�����0����ɶ��˼  order�Ǽ������˼��
	float NCC_max = 0;
	vector<float> NCC_max_data;

	for (int j = 0; j < pyramid_order - 1; ++j)
	{
		pyrDown(inputImage, inputImage, Size(inputImage.cols / 2, inputImage.rows / 2));          //�ȶ�ͼ����и�˹ƽ����Ȼ���ٽ��н�����
	}
	for (int i = 0; i < definition_angle; ++i)
	{
		vector<float> NCC_data_temp = NCC_Matching(inputImage, vect_template_data[i]);

		if (NCC_data_temp.back() > NCC_max)
		{
			NCC_max = NCC_data_temp.back();                     //back�����һλ�� 
			NCC_max_data.clear();
			NCC_max_data.push_back(pow(2, pyramid_order - 1) * (NCC_data_temp[0]));        //���� Ϊʲô��2����ô��η��������ʲô
			NCC_max_data.push_back(pow(2, pyramid_order - 1) * (NCC_data_temp[1]));
			NCC_max_data.push_back(i);                        
			NCC_max_data.push_back(NCC_max);
			cout << "[x,y] = " << NCC_max_data[0] << "," << NCC_max_data[1] << endl;
			cout << "Angle = " << NCC_max_data[2] << endl;               
			cout << "NCC_max = " << NCC_max_data[3] << endl;
		}
	}

	return NCC_max_data;

}

vector<template_data> pyramid_init(Mat templateImage, int definition_angle = 360, int pyramid_order = 7)                 //û̫�� �������Ǹ��Ƕȵ���ɶ����
{
	vector<template_data> vect_template_data;
	template_data template_data_temp;
	Mat rotatedImage;
	for (int i = 0; i <= definition_angle; ++i)                   //++i��ɶ
	{
		rotatedImage = rotate_Image(templateImage, i);
		for (int j = 0; j < pyramid_order - 1; ++j)
		{
			pyrDown(rotatedImage, rotatedImage, Size(rotatedImage.cols / 2, rotatedImage.rows / 2));               //�о�����û�õ�ѭ��j
		}
		NCC_Matching_Init(template_data_temp, rotatedImage, pyramid_order);
		vect_template_data.push_back(template_data_temp);
	}

	return vect_template_data;
}

Mat rotate_Image(Mat inputImage, float angle)
{
	float radian = (float)(angle / 180.0 * CV_PI);

	//���ͼ��
	int maxBorder = (int)(max(inputImage.cols, inputImage.rows) * 1.414); //��Ϊsqrt(2)*max
	int dx = (maxBorder - inputImage.cols) / 2;
	int dy = (maxBorder - inputImage.rows) / 2;
	Mat outputImage;
	copyMakeBorder(inputImage, outputImage, dy, dy, dx, dx, BORDER_CONSTANT);

	//��ת
	Point2f center((float)(outputImage.cols / 2), (float)(outputImage.rows / 2));
	Mat affine_matrix = getRotationMatrix2D(center, angle, 1.0);//�����ת����
	warpAffine(outputImage, outputImage, affine_matrix, outputImage.size());

	//����ͼ����ת֮�����ͼ������ľ���
	float sinVal = abs(sin(radian));
	float cosVal = abs(cos(radian));
	Size targetSize((int)(inputImage.cols * cosVal + inputImage.rows * sinVal),
		(int)(inputImage.cols * sinVal + inputImage.rows * cosVal));

	//��������߿�
	int x = (outputImage.cols - targetSize.width) / 2;
	int y = (outputImage.rows - targetSize.height) / 2;
	Rect rect(x, y, targetSize.width, targetSize.height);
	outputImage = outputImage(rect);
	return outputImage;
}

int main(int argc, char** argv)
{
	Mat templateImg = imread("../image/pattern.bmp");
	cvtColor(templateImg, templateImg, COLOR_RGB2GRAY);                                     //������� ȡtemplateImg��ͼ��õ����ĻҶ�ֵͼ������ΪtemplateImg��
	vector<template_data> vect_template_data = pyramid_init(templateImg, 0, 5);             //����

	Mat input_Image = imread("../image/1.bmp");
	cvtColor(input_Image, input_Image, COLOR_RGB2GRAY);

	//��ʱ
	double time_consumed = static_cast<double>(cv::getTickCount());            //�����ⲻ��

	vector<float> NCC_max_data = pyramid_matching(input_Image, vect_template_data);    //����      
	time_consumed = (cv::getTickCount() - time_consumed) / cv::getTickFrequency();
	cout << "��ʱ��" << time_consumed << " s" << endl;

	float pos_x = NCC_max_data[0];
	float pos_y = NCC_max_data[1];
	float angle = (float)(NCC_max_data[2] / 180.0 * CV_PI);
	float width = templateImg.cols;
	float height = templateImg.rows;
	
	Point p1 = Point(pos_x, pos_y);                                                 //���ǻ���б�ı߿���
	Point p2 = Point(pos_x + width*cos(angle), pos_y - width*sin(angle));
	Point p3 = Point(pos_x - height*sin(angle) + width*cos(angle), pos_y + height*cos(angle) - width*sin(angle));
	Point p4 = Point(pos_x + height*sin(angle), pos_y + height*cos(angle));

	line(input_Image, p1, p2, Scalar(255), 2);                                        
	line(input_Image, p2, p3, Scalar(255), 2);
	line(input_Image, p3, p4, Scalar(255), 2);
	line(input_Image, p4, p1, Scalar(255), 2);

	cout << "[x,y] = " << NCC_max_data[0] << " , "<<NCC_max_data[1] << endl;
	cout << "Angle = " << NCC_max_data[2] << endl;
	cout << "NCC_max = " << NCC_max_data[3] << endl;
	imshow("output", input_Image);
	waitKey(0);

	return 0;
}