/* FAST corner detector algorithm implementation.
 * @file
 * @date 2018-10-16
 * @author Anonymous
 */

#include "cvlib.hpp"

#include <ctime>
#include <fstream>

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

void dbg_dump_brief_pairs(std::vector<std::pair<cv::Point2i, cv::Point2i>> pairs)
{
    std::ofstream file("brief_pairs.txt");
    for (auto pair : pairs)
        file << pair.first.x << " " << pair.first.y << " " << pair.second.x << " " << pair.second.y << "\n";
}

void corner_detector_fast::initBriefPairs()
{
    const size_t amount = dwords * 32;
    const int radius = 15;

    auto uniform_rand = []() { return float(rand() % 10001) / 10000.0f; };
    auto gen_coord = [radius, uniform_rand]() -> float { return tanh(3.5f * (uniform_rand() - 0.5f)) * radius; };

    brief_pairs.reserve(amount);
    for (size_t i = 0; i < amount; ++i)
        brief_pairs.emplace_back(std::make_pair(cv::Point2f(gen_coord(), gen_coord()), cv::Point2f(gen_coord(), gen_coord())));
}

float getDirectionRadians(float index)
{
    // 0 - north
    // pi/2 - east
    // pi - south
    // 3*pi/2 - west
    return 2 * float(M_PI) * index / 16.0f;
}

static inline int soft_sign(int diff, int brightness_threshold)
{
    if (diff > brightness_threshold)
        return 1;
    else if (diff < -brightness_threshold)
        return -1;
    else
        return 0;
}

bool corner_detector_fast::testPixel(cv::Mat& image, cv::Point2i point, float& direction)
{
    static const cv::Point2i test_points_relative[16] = {{0, -3}, {1, -3}, {2, -2}, {3, -1}, {3, 0},  {3, 1},   {2, 2},   {1, 3},
                                                         {0, 3},  {-1, 3}, {-2, 2}, {-3, 1}, {-3, 0}, {-3, -1}, {-2, -2}, {-1, -3}};

    int begin_sign = 0, current_sign = 0;
    size_t begin_count = 0, count = 0, first_index = 0;

    int diff = (int)(image.at<uint8_t>(point + test_points_relative[0])) - (int)(image.at<uint8_t>(point));
    int sign = soft_sign(diff, brightness_threshold);
    bool begin_flag = sign != 0;

    for (size_t i = 1; i < 16; ++i)
    {
        diff = (int)(image.at<uint8_t>(point + test_points_relative[i])) - (int)(image.at<uint8_t>(point));
        sign = soft_sign(diff, brightness_threshold);

        if (sign == current_sign)
        {
            count++;
        }
        else
        {
            if (current_sign != 0 && count >= succeded_points_threshold)
            {
                direction = getDirectionRadians(0.5f * (i - 1 + first_index));
                return true;
            }

            if (begin_flag)
            {
                begin_count = count;
                begin_sign = current_sign;
                begin_flag = false;
            }

            first_index = i;
            count = 1;

            current_sign = sign;
        }
    }

    if (current_sign != 0)
    {
        if (current_sign == begin_sign && count + begin_count >= succeded_points_threshold)
        {
            float index = 0.5f * (first_index + (begin_count - 1 + 16));
            direction = getDirectionRadians(index >= 16.0f ? index - 16.0f : index);
            return true;
        }
        else if (count >= succeded_points_threshold)
        {
            direction = getDirectionRadians(0.5f * (15 + first_index));
            return true;
        }
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
    cv::Mat image;

    if (input.channels() == 1)
        input.copyTo(image);
    else if (input.channels() == 3)
        cv::cvtColor(input, image, cv::COLOR_BGR2GRAY);

    cv::GaussianBlur(image, image, cv::Size(9, 9), 2.0, 2.0, cv::BORDER_CONSTANT);

    const int desc_length = static_cast<int>(brief_pairs.size()) / 32;
    descriptors.create(static_cast<int>(keypoints.size()), desc_length, CV_32S);
    auto desc_mat = descriptors.getMat();

    int* ptr = reinterpret_cast<int*>(desc_mat.ptr());
    for (int k = 0; k < desc_mat.rows; ++k)
    {
        cv::KeyPoint& pt = keypoints[k];
        // float cosa = std::cos(pt.angle);
        // float sina = std::sin(pt.angle);

        uint32_t val = 0;
        for (int i = 0; i < desc_length; ++i)
        {
            for (int p = 32 * i; p < 32 * i + 32; ++p)
            {
                auto& pair = brief_pairs[p];
                // cv::Point2f new_first = pt.pt + rotatePoint(pair.first, cosa, sina);
                // cv::Point2f new_second = pt.pt + rotatePoint(pair.second, cosa, sina);
                cv::Point2f new_first = pt.pt + pair.first;
                cv::Point2f new_second = pt.pt + pair.second;
                int first_val = isInsideImage(image, new_first) ? image.at<uint8_t>(new_first) : 0;
                int second_val = isInsideImage(image, new_second) ? image.at<uint8_t>(new_second) : 0;
                val = (val << 1) | (std::abs(second_val - first_val) >= descriptor_threshold);
            }
            desc_mat.at<int32_t>(k, i) = val;
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
