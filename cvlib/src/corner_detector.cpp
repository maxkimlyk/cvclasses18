/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <ctime>

namespace cvlib
{
// static
cv::Ptr<corner_detector_fast> corner_detector_fast::create()
{
    return cv::makePtr<corner_detector_fast>();
}

bool corner_detector_fast::testPixel(cv::Mat& image, cv::Point2i point)
{
    static const cv::Point2i test_points_relative[16] = {{0, -3}, {1, -3}, {2, -2}, {3, -1}, {3, 0},  {3, 1},   {2, 2},   {1, 3},
                                                         {0, 3},  {-1, 3}, {-2, 2}, {-3, 1}, {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3}};

    bool first_flag = true;
    size_t first_count = 0, count = 0;

    for (size_t i = 0; i < 16; ++i)
    {
        if (std::abs((int)(image.at<uint8_t>(point + test_points_relative[i])) - (int)(image.at<uint8_t>(point))) >= brightness_threshold)
            ++count;
        else
        {
            if (count >= succeded_points_threshold)
                return true;
            if (first_flag)
            {
                first_count = count;
                first_flag = false;
            }
            count = 0;
        }
    }

    return count + first_count >= succeded_points_threshold;
}

void corner_detector_fast::detect(cv::InputArray input, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray input_mask)
{
    keypoints.clear();
    cv::Mat image = input.getMat(), mask = input_mask.getMat();
    for (int i = 3; i < image.rows - 3; ++i)
        for (int j = 3; j < image.cols - 3; ++j)
            if ((mask.empty() || mask.at<uint8_t>(j, i)) && testPixel(image, cv::Point2i{j, i}))
                keypoints.emplace_back(cv::KeyPoint(float(j), float(i), 1.0f));
}

void corner_detector_fast::compute(cv::InputArray, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    std::srand(unsigned(std::time(0))); // \todo remove me
    // \todo implement any binary descriptor
    const int desc_length = 2;
    descriptors.create(static_cast<int>(keypoints.size()), desc_length, CV_32S);
    auto desc_mat = descriptors.getMat();
    desc_mat.setTo(0);

    int* ptr = reinterpret_cast<int*>(desc_mat.ptr());
    for (const auto& pt : keypoints)
    {
        for (int i = 0; i < desc_length; ++i)
        {
            *ptr = std::rand();
            ++ptr;
        }
    }
}

void corner_detector_fast::detectAndCompute(cv::InputArray, cv::InputArray, std::vector<cv::KeyPoint>&, cv::OutputArray descriptors, bool /*= false*/)
{
    // \todo implement me
}
} // namespace cvlib
