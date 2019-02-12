#include <iostream>
#include <opencv2/opencv.hpp>

int main()
{
    cv::Mat src_image = cv::imread("../data/image01.png");
    cv::namedWindow("window", cv::WINDOW_NORMAL);
    cv::imshow("window", src_image);
    cv::waitKey(0);
}