#include <iostream>
#include <opencv2/opencv.hpp>

class DescriptorWithID
{
public:
    DescriptorWithID(cv::Ptr<cv::FeatureDetector> detector)
        : m_descriptor_size(detector->descriptorSize()),
          m_descriptor_type(detector->descriptorType())
    {
        m_descriptors = cv::Mat(0, m_descriptor_size, m_descriptor_type);
        m_ids = cv::Mat(0, 1, CV_32SC1);
    }

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

private:
    int m_descriptor_size;
    int m_descriptor_type;
};

//class GriddedDescriptors
//{
//public:
//    GriddedDescriptors(cv::Size size, cv::Ptr<cv::Feature2D> detector, cv::Mat descriptors, int grid_size = 10)
//        : m_grid_size(grid_size), m_row_blocks(size.width / grid_size), m_col_blocks(size.height / grid_size)
//    {
//        // グリッドに分解
//    }
//
//private:
//    const int m_grid_size;  //[pixel]
//    const int m_row_blocks;
//    const int m_col_blocks;
//
//
//};

int main(int argc, char* argv[])
{
    cv::Mat src_image = cv::imread("../data/image01.png");
    int grid_size = 40;
    if (argc == 2)
        grid_size = std::atoi(argv[1]);

    cv::Mat image1 = src_image.clone();
    cv::Mat image2 = src_image.clone();  // とりあえず同じ画像

    // Epipolar関係
    cv::Mat E = (cv::Mat_<double>(3, 3) << 0, 0, 0, 0, 0, -1, 0, 1, 0);  // x軸方向に移動した
    cv::Mat K = cv::Mat::eye(3, 3, CV_64FC1);
    cv::Mat F = K.inv().t() * E * K.inv();
    std::cout << "F\n"
              << F << std::endl;

    // 特徴量検出・記述
    cv::Ptr<cv::FeatureDetector> detector = cv::AKAZE::create();
    std::vector<cv::KeyPoint> keypoints1, keypoints2;
    cv::Mat descriptors1, descriptors2;
    detector->detectAndCompute(image1, cv::noArray(), keypoints1, descriptors1);
    detector->detectAndCompute(image2, cv::noArray(), keypoints2, descriptors2);

    // グリッド分割(とりあえず一方向だけ)
    int grid_width = std::ceil(1.0 * image2.size().width / grid_size);
    int grid_height = std::ceil(1.0 * image2.size().height / grid_size);
    std::vector<DescriptorWithID> grid_elements2(grid_width * grid_height, DescriptorWithID(detector));
    for (size_t i = 0; i < descriptors2.rows; i++) {
        cv::KeyPoint key = keypoints2.at(i);
        cv::Mat des = descriptors2.row(i);

        int grid_no = (key.pt.x / grid_size) + (key.pt.y / grid_size) * grid_width;
        grid_elements2.at(grid_no).push_back(i, des);
    }
    std::cout << grid_height << " " << grid_width << std::endl;

    // Matching each keypoints
    cv::BFMatcher matcher(cv::NORM_HAMMING);
    std::vector<std::vector<cv::DMatch>> knn_matches;
    std::vector<cv::DMatch> matches;

    for (size_t i = 0; i < descriptors1.rows; i++) {
        cv::Point2f tmp = keypoints1.at(i).pt;
        cv::Mat pt = (cv::Mat_<double>(3, 1) << tmp.x, tmp.y, 1);
        cv::Mat des = descriptors1.row(i);

        DescriptorWithID merged(detector);

        // Epiline
        cv::Mat line = pt.t() * F;
        double a = line.at<double>(0);
        double b = line.at<double>(1);
        double c = line.at<double>(2);
        double norm = std::sqrt(a * a + b * b);

        // Merge
        for (size_t w = 0; w < grid_width; w++) {
            for (size_t h = 0; h < grid_height; h++) {
                double product = a * (w + 0.5) * grid_size + b * (h + 0.5) * grid_size + c;
                if (std::abs(product) < grid_size * norm) {  // epilineから格子中心までの距離が格子間隔よりも小さければ併合
                    merged.merge(grid_elements2.at(w + h * grid_width));
                }
            }
        }
        // std::cout << "id: " << i << " has " << merged.size() << " candidates" << std::endl;

        // Matching
        std::vector<std::vector<cv::DMatch>> tmp_matches;
        matcher.knnMatch(des, merged.m_descriptors, tmp_matches, 1);  // 1対多のマッチング

        // Trasnlate
        int query = tmp_matches.at(0).at(0).queryIdx;
        int train = tmp_matches.at(0).at(0).trainIdx;
        float distance = tmp_matches.at(0).at(0).distance;
        matches.push_back(cv::DMatch(i, merged.m_ids.at<int>(train), distance));
    }


    cv::Mat show_image;
    drawMatches(image1, keypoints1, image2, keypoints2, matches, show_image, cv::Scalar::all(-1),
        cv::Scalar::all(-1), std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);
    cv::imshow("window", show_image);


    //cv::namedWindow("window", cv::WINDOW_NORMAL);
    //cv::imshow("window", src_image);
    cv::waitKey(0);
}