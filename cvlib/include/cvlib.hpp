/* Computer Vision Functions.
 * @file
 * @date 2018-09-10
 * @author Max Kimlyk
 */

#ifndef __CVLIB_HPP__
#define __CVLIB_HPP__

#include <opencv2/opencv.hpp>

#ifndef M_PI
#define M_PI 3.1415926535
#endif

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

/// \brief FAST corner detection algorithm
class corner_detector_fast : public cv::Feature2D
{
public:
    corner_detector_fast();

    /// \brief Fabrique method for creating FAST detector
    static cv::Ptr<corner_detector_fast> create();

    /// \see Feature2d::detect
    virtual void detect(cv::InputArray image, CV_OUT std::vector<cv::KeyPoint>& keypoints, cv::InputArray mask = cv::noArray()) override;

    /// \brief Set points threshold
    inline void setSuccededPointsThreshold(size_t thresh)
    {
        succeded_points_threshold = thresh > 4 ? thresh : 4;
    }

    // \brief Set brightness threshold
    inline void setBrightnessTreshold(size_t thresh)
    {
        brightness_threshold = thresh;
    }

    /// \see Feature2d::compute
    virtual void compute(cv::InputArray image, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors) override;

    /// \see Feature2d::detectAndCompute
    virtual void detectAndCompute(cv::InputArray image, cv::InputArray mask, std::vector<cv::KeyPoint>& keypoints, cv::OutputArray descriptors,
                                  bool useProvidedKeypoints = false) override;

    /// \see Feature2d::getDefaultName
    virtual cv::String getDefaultName() const override
    {
        return "FAST_Binary";
    }

private:
    size_t succeded_points_threshold = 12;
    size_t brightness_threshold = 40;
    size_t descriptor_threshold = 40;
    std::vector<std::pair<cv::Point2i, cv::Point2i>> brief_pairs;

    /// \brief Test pixel whether it is corner or not
    bool testPixel(cv::Mat& image, cv::Point2i point, float &direction);

    void initBriefPairs();
};
} // namespace cvlib

#endif // __CVLIB_HPP__
