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

corner_detector_fast::corner_detector_fast()
{
    initBriefPairs();
}

void corner_detector_fast::initBriefPairs()
{
    const size_t amount = 128;
    const int radius = 10;

    auto gen_coord = [radius]() -> float { return float(rand() % (2 * radius + 1) - radius); };

    brief_pairs.reserve(amount);
    for (size_t i = 0; i < amount; ++i)
    {
        brief_pairs.emplace_back(std::make_pair(cv::Point2f(gen_coord(), gen_coord()), cv::Point2f(gen_coord(), gen_coord())));
    }
}

float getDirectionRadians(float index)
{
    // 0 - north
    // pi/2 - east
    // pi - south
    // 3*pi/2 - west
    return 2 * float(M_PI) * index / 16.0f;
}

bool corner_detector_fast::testPixel(cv::Mat& image, cv::Point2i point, float& direction)
{
    static const cv::Point2i test_points_relative[16] = {{0, -3}, {1, -3}, {2, -2}, {3, -1}, {3, 0},  {3, 1},   {2, 2},   {1, 3},
                                                         {0, 3},  {-1, 3}, {-2, 2}, {-3, 1}, {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3}};

    bool first_flag = true;
    size_t first_count = 0, count = 0, first_index = 0;

    for (size_t i = 0; i < 16; ++i)
    {
        if (std::abs((int)(image.at<uint8_t>(point + test_points_relative[i])) - (int)(image.at<uint8_t>(point))) >= brightness_threshold)
        {
            if (count++ == 0)
                first_index = i;
        }
        else
        {
            if (count >= succeded_points_threshold)
            {
                direction = getDirectionRadians(0.5f * (i - 1 + first_index));
                return true;
            }
            if (first_flag)
            {
                first_count = count;
                first_flag = false;
            }
            count = 0;
            first_index = 0;
        }
    }

    if (count + first_count >= succeded_points_threshold)
    {
        float index = 0.5f * (first_index + (first_count - 1 + 16));
        direction = getDirectionRadians(index >= 16.0f ? index - 16.0f : index);
        return true;
    }

    return false;
}

cv::Mat make_grayscale(const cv::Mat& input)
{
    cv::Mat gray;
    cv::cvtColor(input, gray, cv::COLOR_BGR2GRAY);
    return gray;
}

void corner_detector_fast::detect(cv::InputArray input, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray input_mask)
{
    keypoints.clear();
    cv::Mat image = input.getMat(), mask = input_mask.getMat();
    if (image.channels() != 1)
        cv::cvtColor(image, image, cv::COLOR_BGR2GRAY);

    for (int i = 3; i < image.rows - 3; ++i)
        for (int j = 3; j < image.cols - 3; ++j)
        {
            float dir;
            if ((mask.empty() || mask.at<uint8_t>(j, i)) && testPixel(image, cv::Point2i{j, i}, dir))
                keypoints.emplace_back(cv::KeyPoint(float(j), float(i), 1.0f, dir));
        }
}

inline cv::Point2f rotatePoint(cv::Point2f pt, float cosa, float sina)
{
    return cv::Point2f(std::round(pt.x * cosa - pt.y * sina), std::round(pt.x * sina + pt.y * cosa));
}

inline bool isInsideImage(const cv::Mat& image, cv::Point2f pt)
{
    return 0 <= pt.x && pt.x < float(image.cols) - 0.5 && 0 <= pt.y && pt.y < float(image.rows) - 0.5;
}

void corner_detector_fast::compute(cv::InputArray input, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors)
{
    cv::Mat image = input.getMat();

    const int desc_length = static_cast<int>(brief_pairs.size()) / 32;
    descriptors.create(static_cast<int>(keypoints.size()), desc_length, CV_32S);
    auto desc_mat = descriptors.getMat();
    desc_mat.setTo(0);

    int* ptr = reinterpret_cast<int*>(desc_mat.ptr());
    for (const auto& pt : keypoints)
    {
        float cosa = std::cos(pt.angle);
        float sina = std::sin(pt.angle);

        uint32_t val = 0;
        for (int i = 0; i < desc_length; ++i)
        {
            for (int p = 32 * i; p < 32 * i + 32; ++p)
            {
                auto& pair = brief_pairs[p];
                cv::Point2f new_first = pt.pt + rotatePoint(pair.first, cosa, sina);
                cv::Point2f new_second = pt.pt + rotatePoint(pair.second, cosa, sina);
                int first_val = isInsideImage(image, new_first) ? image.at<uint8_t>(new_first) : 0;
                int second_val = isInsideImage(image, new_second) ? image.at<uint8_t>(new_second) : 0;
                val = (val << 1) | (std::abs(second_val - first_val) >= descriptor_threshold);
            }
            *ptr = val;
            ++ptr;
        }
    }
}

void corner_detector_fast::detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints,
                                            cv::OutputArray descriptors, bool useProvidedKeypoints)
{
    detect(image, keypoints, mask);
    compute(image, keypoints, descriptors);
}
} // namespace cvlib
