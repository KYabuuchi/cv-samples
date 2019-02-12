// Epipolor拘束を利用して，特徴点のマッチングを拘束に行う
// 画像間のF行列が必要
// 画像を格子状に分割し，Epilineに近いブロックに含まれる特徴点のみを収集しマッチングをする
// NOTE: 画像サイズが2枚とも同じであることを仮定している
// NOTE: F行列がわかる環境下で画像サイズがことなるようなシチュエーションは少ない
#include "config.hpp"
#include "util.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>

class DescriptorWithID
{
public:
    DescriptorWithID(cv::Ptr<cv::FeatureDetector> detector)
        : m_descriptors(0, detector->descriptorSize(), detector->descriptorType()),
          m_ids(0, 1, CV_32SC1) {}

    size_t size() const
    {
        return m_descriptors.rows;
    }

    void push_back(int id, cv::Mat descriptor)
    {
        cv::vconcat(m_ids, id * cv::Mat::ones(1, 1, CV_32SC1), m_ids);
        m_descriptors.push_back(descriptor);
    }

    void merge(DescriptorWithID other)
    {
        m_descriptors.push_back(other.m_descriptors);
        m_ids.push_back(other.m_ids);
    }

    cv::Mat m_descriptors;
    cv::Mat m_ids;
};

class MatcherByEpipolar
{
public:
    MatcherByEpipolar(
        cv::Ptr<cv::FeatureDetector> detector,
        cv::Ptr<cv::DescriptorMatcher> matcher,
        cv::Size size,
        cv::Size grid_size = cv::Size(64, 48))
        : m_F(cv::Mat::eye(3, 3, CV_64FC1)),
          m_detector(detector), m_matcher(matcher),
          m_size(size), m_grid_size(grid_size),
          m_grid_num(cv::Size(
              static_cast<int>(std::ceil(1.0 * size.width / grid_size.width)),
              static_cast<int>(std::ceil(1.0 * size.height / grid_size.height))))
    {
        std::cout << "Image : " << m_size << std::endl;
        std::cout << "Grid  : " << m_grid_num << std::endl;
        std::cout << "Grid Size: " << m_grid_size << std::endl;
    }

    cv::Mat m_F;
    cv::Ptr<cv::FeatureDetector> m_detector;
    cv::Ptr<cv::DescriptorMatcher> m_matcher;
    const cv::Size m_size;
    const cv::Size m_grid_size;
    const cv::Size m_grid_num;

    // Epipolar拘束を利用して双方向マッチングする(1->2->1)
    std::vector<cv::DMatch> matching(
        std::vector<cv::KeyPoint> keypoints1, cv::Mat descriptors1,
        std::vector<cv::KeyPoint> keypoints2, cv::Mat descriptors2,
        cv::Mat F)
    {
        m_F = F;

        // Gridding
        m_gridded_elements2 = std::vector<DescriptorWithID>(m_grid_num.width * m_grid_num.height, DescriptorWithID(m_detector));
        for (int i = 0; i < descriptors2.rows; i++) {
            cv::KeyPoint key = keypoints2.at(i);
            cv::Mat des = descriptors2.row(i);

            int grid_no = static_cast<int>(key.pt.x / static_cast<float>(m_grid_size.width))
                          + static_cast<int>(key.pt.y / static_cast<float>(m_grid_size.height)) * m_grid_num.width;
            m_gridded_elements2.at(grid_no).push_back(i, des);
        }

        // Matching each keypoints
        std::vector<cv::DMatch> matches;
        for (int i = 0; i < descriptors1.rows; i++) {
            cv::Point2f tmp = keypoints1.at(i).pt;
            cv::Mat x1 = (cv::Mat_<double>(3, 1) << tmp.x, tmp.y, 1);
            cv::Mat des = descriptors1.row(i);

            DescriptorWithID merged = mergeGridsByEpiline(x1, m_F);

            // Matching
            std::vector<std::vector<cv::DMatch>> tmp_matches;
            m_matcher->knnMatch(des, merged.m_descriptors, tmp_matches, 1);
            if (tmp_matches.at(0).empty())
                continue;

            // push back
            int train = tmp_matches.at(0).at(0).trainIdx;
            float distance = tmp_matches.at(0).at(0).distance;
            matches.push_back(cv::DMatch(i, merged.m_ids.at<int>(train), distance));
        }

        return matches;
    }

private:
    std::vector<DescriptorWithID> m_gridded_elements1;
    std::vector<DescriptorWithID> m_gridded_elements2;

    DescriptorWithID mergeGridsByEpiline(cv::Mat x1, cv::Mat F)
    {
        DescriptorWithID merged(m_detector);

        // Epiline
        cv::Mat line = x1.t() * F;
        double a = line.at<double>(0);
        double b = line.at<double>(1);
        double c = line.at<double>(2);
        double norm = std::sqrt(a * a + b * b);

        // Merge
        for (int w = 0; w < m_grid_num.width; w++) {
            for (int h = 0; h < m_grid_num.height; h++) {

                // epilineから格子中心までの距離が格子間隔よりも小さければ併合
                double product = a * (w + 0.5) * m_grid_size.width + b * (h + 0.5) * m_grid_size.height + c;  // (ax+by+x)/sqrt(aa+bb) < grid_size
                if (std::abs(product) < std::max(m_grid_size.width, m_grid_size.height) * norm) {
                    merged.merge(m_gridded_elements2.at(w + h * m_grid_num.width));
                }
            }
        }

        return merged;
    }
};


int main(int argc, char** argv)
{
    cv::Size grid_size(64, 48);
    if (argc == 2) {
        double gain = std::atof(argv[1]);
        grid_size.width = static_cast<int>(gain * grid_size.width);
        grid_size.height = static_cast<int>(gain * grid_size.height);
    }

    // Load Image
    cv::Mat image1, image2;
    if (not Util::readStereoImage("../data/stereo01.png", image1, image2))
        return -1;
    std::cout << "Image size: " << image1.size() << " " << image2.size() << std::endl;

    // Fundamental Matrix
    cv::Mat K = Params::ZED_INTRINSIC;
    cv::Mat T = Params::ZED_EXTRINSIC;
    cv::Mat E = Util::calcEssentialFromRt(T.colRange(0, 3), T.col(3));
    cv::Mat F = K.inv().t() * E * K.inv();
    std::cout << "Fundamental Matrix\n"
              << F << std::endl;

    // detector & matcher
    cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(cv::NORM_HAMMING);
    std::cout << "detector: " << detector->getDefaultName() << std::endl;

    // Detect & Descript Features
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    detector->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);

    // Matching by using Epipolar Constrant
    MatcherByEpipolar epi_matcher(detector, matcher, image1.size(), grid_size);
    std::vector<cv::DMatch> matches = epi_matcher.matching(
        keypoints1, descriptors1,
        keypoints2, descriptors2, F);

    // Show
    cv::Mat show_image;
    drawMatches(image1, keypoints1, image2, keypoints2, matches, show_image, cv::Scalar::all(-1),
        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::namedWindow("window", CV_WINDOW_NORMAL);
    cv::resizeWindow("window", cv::Size(1280, 480));
    cv::imshow("window", show_image);
    cv::waitKey(0);
}