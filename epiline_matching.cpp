// Epipolor拘束を利用して，特徴点のマッチングを拘束に行う
// 画像間のF行列が必要
// 画像を格子状に分割し，Epilineに近いブロックに含まれる特徴点のみを収集しマッチングをする
// NOTE: 画像サイズが2枚とも同じであることを仮定している
// NOTE: F行列がわかる環境下で画像サイズがことなるようなシチュエーションは少ない
// TODO: むしろ遅いので，cv::Matをコピーしないで済む実装をする
#include "config.hpp"
#include "util.hpp"
#include <iostream>
#include <opencv2/opencv.hpp>


class MatcherByEpipolar
{
public:
    // 検出器
    // 対応器
    // 画像サイズ
    // 格子のサイズ
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
        std::cout << "Grid  : " << m_grid_num << std::endl;
        std::cout << "Grid Size: " << m_grid_size << std::endl;
    }

    // public member
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

        cv::TickMeter tm;
        tm.start();

        std::cout << "hello" << std::endl;
        // Gridding
        m_gridded_elements2.clear();
        m_gridded_elements2.resize(m_grid_num.width * m_grid_num.height);
        for (int i = 0; i < descriptors2.rows; i++) {
            cv::KeyPoint key = keypoints2.at(i);
            cv::Mat des = descriptors2.row(i);

            int grid_no = static_cast<int>(key.pt.x / static_cast<float>(m_grid_size.width))
                          + static_cast<int>(key.pt.y / static_cast<float>(m_grid_size.height)) * m_grid_num.width;
            m_gridded_elements2.at(grid_no).push_back(i);
        }
        tm.stop();
        std::cout << tm.getTimeSec() << std::endl;

        // Matching each keypoints
        std::vector<cv::DMatch> matches;
        for (int i = 0; i < descriptors1.rows; i++) {
            cv::Point2f tmp = keypoints1.at(i).pt;
            cv::Mat x1 = (cv::Mat_<double>(3, 1) << tmp.x, tmp.y, 1);
            cv::Mat des = descriptors1.row(i);

            tm.reset();
            tm.start();
            cv::Mat mask = mergeGridsByEpiline(x1, descriptors2.rows);
            tm.stop();
            std::cout << "G " << tm.getTimeSec() << std::endl;

            tm.reset();
            tm.start();
            // Matching
            std::vector<std::vector<cv::DMatch>> tmp_matches;
            m_matcher->knnMatch(des, descriptors2, tmp_matches, 1, mask);
            tm.stop();
            std::cout << "M " << tm.getTimeSec() << std::endl;
            if (tmp_matches.at(0).empty())
                continue;

            // push back
            int train = tmp_matches.at(0).at(0).trainIdx;
            float distance = tmp_matches.at(0).at(0).distance;
            matches.push_back(cv::DMatch(i, train, distance));
        }

        return matches;
    }

private:
    std::vector<std::vector<size_t>> m_gridded_elements1;
    std::vector<std::vector<size_t>> m_gridded_elements2;

    cv::Mat mergeGridsByEpiline(cv::Mat x1, size_t size)
    {
        cv::Mat mask = cv::Mat::zeros(1, size, CV_8UC1);
        // Epiline
        cv::Mat line = x1.t() * m_F;
        double a = line.at<double>(0);
        double b = line.at<double>(1);
        double c = line.at<double>(2);
        double square_norm = m_grid_size.width * m_grid_size.width * (a * a + b * b);

        int test = 0;
        for (int w = 0; w < m_grid_num.width; w++) {
            for (int h = 0; h < m_grid_num.height; h++) {
                // epilineから格子中心までの距離が格子間隔よりも小さければ併合
                double product = a * (w + 0.5) * m_grid_size.width + b * (h + 0.5) * m_grid_size.height + c;

                // (ax+by+c)/sqrt(aa+bb) < grid_size => (ax+by+c)^2  < grid_size^2 * (aa+bb)
                if (product * product < square_norm) {
                    std::vector<size_t> candidates = m_gridded_elements2.at(w + h * m_grid_num.width);
                    for (const size_t& c : candidates) {
                        mask.at<unsigned char>(0, c) = 1;
                        test++;
                    }
                }
            }
        }
        // std::cout << test << std::endl;
        return mask;
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
    std::cout << "Fundamental Matrix:\n"
              << F << std::endl;

    // detector & matcher
    cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create();
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::BFMatcher::create(detector->defaultNorm());
    std::cout << "detector: " << detector->getDefaultName() << std::endl;

    // Detect & Descript Features
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    detector->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);
    std::cout << "feature size " << keypoints1.size() << std::endl;

    {
        // timer
        cv::TickMeter tm;
        tm.start();

        std::vector<std::vector<cv::DMatch>> knn_matches;
        matcher->knnMatch(descriptors1, descriptors2, knn_matches, 2);

        const float r = 0.7f;
        std::vector<cv::DMatch> matches;
        for (size_t i = 0; i < knn_matches.size(); i++) {
            if (knn_matches[i].size() < 2) {
                continue;
            }

            if (knn_matches[i][0].distance < r * knn_matches[i][1].distance) {
                matches.push_back(knn_matches[i][0]);
            }
        }

        // timer
        tm.stop();
        std::cout << "\ntime: " << tm.getTimeSec() << std::endl;

        // Show
        cv::Mat show_image;
        drawMatches(image1, keypoints1, image2, keypoints2, matches, show_image, cv::Scalar::all(-1),
            cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::namedWindow("window1", CV_WINDOW_NORMAL);
        cv::resizeWindow("window1", cv::Size(1280, 480));
        cv::imshow("window1", show_image);
    }

    {
        // timer
        cv::TickMeter tm;
        tm.start();

        // Matching by using Epipolar Constrant
        MatcherByEpipolar epi_matcher(detector, matcher, image1.size(), grid_size);
        std::vector<cv::DMatch> matches = epi_matcher.matching(
            keypoints1, descriptors1,
            keypoints2, descriptors2, F);

        // timer
        tm.stop();
        std::cout << "\ntime: " << tm.getTimeSec() << std::endl;

        // Show
        cv::Mat show_image;
        drawMatches(image1, keypoints1, image2, keypoints2, matches, show_image, cv::Scalar::all(-1),
            cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
        cv::namedWindow("window", CV_WINDOW_NORMAL);
        cv::resizeWindow("window", cv::Size(1280, 480));
        cv::imshow("window", show_image);
    }

    cv::waitKey(0);
}