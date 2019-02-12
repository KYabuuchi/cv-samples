#pragma once
#include <opencv2/opencv.hpp>

namespace Util
{

// ステレオ画像を分割
bool readStereoImage(std::string path, cv::Mat& image1, cv::Mat& image2)
{
    cv::Mat src_image = cv::imread(path, cv::IMREAD_UNCHANGED);
    if (src_image.empty())
        return false;

    int half = src_image.cols / 2;
    image1 = src_image.colRange(0, half);
    image2 = src_image.colRange(half, half + half);

    return true;
}

}  // namespace Util