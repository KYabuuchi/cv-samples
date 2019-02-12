#pragma once
#include <opencv2/core/core.hpp>

namespace Params
{
const cv::Mat ZED_INTRINSIC = (cv::Mat_<double>(3, 3) << 350, 0, 336, 0, 350, 336, 0, 0, 1);
const cv::Mat ZED_EXTRINSIC = (cv::Mat_<double>(3, 4) << 1, 0, 0, 120, 0, 1, 0, 0, 0, 0, 1, 0);
}  // namespace Params