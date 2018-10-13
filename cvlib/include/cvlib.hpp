/* Computer Vision Functions.
 * @file
 * @date 2018-09-10
 * @author Max Kimlyk
 */

#ifndef __CVLIB_HPP__
#define __CVLIB_HPP__

#include <opencv2/opencv.hpp>

namespace cvlib
{
/// \brief Split and merge algorithm for image segmentation
/// \param image, in - input image
/// \param stddev, in - threshold to treat regions as homogeneous
/// \param min_chunk_size, in - chunk of this size will not be splited into smaller chunks
/// \return segmented image
cv::Mat split_and_merge(const cv::Mat& image, double stddev, int min_chunk_size = 10);

/// \brief Segment texuture on passed image according to sample in ROI
/// \param image, in - input image
/// \param roi, in - region with sample texture on passed image
/// \param eps, in - threshold parameter for texture's descriptor distance
/// \return binary mask with selected texture
cv::Mat select_texture(const cv::Mat& image, const cv::Rect& roi, double eps);

/// \brief Motion Segmentation algorithm
class motion_segmentation : public cv::BackgroundSubtractor
{
public:
    /// \brief ctor
	motion_segmentation(const cv::Mat& initial_frame);

    /// \see cv::BackgroundSubtractor::apply
    void apply(cv::InputArray image, cv::OutputArray fgmask, double learningRate = -1) override;

    /// \see cv::BackgroundSubtractor::BackgroundSubtractor
    void getBackgroundImage(cv::OutputArray backgroundImage) const override;

    /// \brief set variance threshold
    void setVarThreshold(double threshold);

private:
    cv::Mat distribution_means_;
    cv::Mat distribution_var_;
    double variance_threshold_ = 2.5;
};
} // namespace cvlib

#endif // __CVLIB_HPP__
