/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Max Kimlyk
 */

#include "cvlib.hpp"

#include <iostream>

namespace cvlib
{
const float START_VAR = 100;

motion_segmentation::motion_segmentation(const cv::Mat& initial_frame):
    distribution_means_(initial_frame),
    distribution_var_(std::vector<int> {initial_frame.rows, initial_frame.cols}, CV_32FC1, START_VAR)
{
    distribution_means_.convertTo(distribution_means_, CV_32FC1);
}

void motion_segmentation::apply(cv::InputArray input, cv::OutputArray output, double learning_rate)
{
	cv::Mat image = input.getMat();

    const double p = 0.0 <= learning_rate && learning_rate <= 1.0 ? learning_rate : 0.5;

    cv::Mat result = cv::Mat::zeros(image.size(), CV_8UC1);

    for (int i = 0; i < image.rows; ++i)
        for (int j = 0; j < image.cols; ++j)
        {
            uint8_t value = image.at<uint8_t>(i, j);
            float mean = distribution_means_.at<float>(i, j);
            float var = distribution_var_.at<float>(i, j);

            if (std::abs(value - mean) / sqrt(var) > variance_threshold_)
                result.at<uint8_t>(i, j) = 255;

            float diff = value - distribution_means_.at<float>(i, j);
            float diff2 = diff * diff - distribution_var_.at<float>(i, j);
            distribution_means_.at<float>(i, j) += float(p) * diff;
            distribution_var_.at<float>(i, j) += float(p) * diff2;
        }

    output.assign(result);
}

void motion_segmentation::getBackgroundImage(cv::OutputArray backgroundImage) const
{
    cv::Mat background;
    distribution_means_.convertTo(background, CV_8UC1);
    backgroundImage.assign(background);
}

void motion_segmentation::setVarThreshold(double threshold)
{
    variance_threshold_ = threshold;
}

} // namespace cvlib
