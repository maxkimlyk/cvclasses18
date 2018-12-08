/* Computer Vision Functions.
 * @file
 * @date 2018-09-10
 * @author Max Kimlyk
 */

#ifndef __CVLIB_HPP__
#define __CVLIB_HPP__

#include <iostream>

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
    const int dwords = 4;

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
    bool testPixel(cv::Mat& image, cv::Point2i point, float& direction);

    void initBriefPairs();
};

/// \brief Descriptor matched based on ratio of SSD
class descriptor_matcher : public cv::DescriptorMatcher
{
    public:
    /// \brief ctor
    descriptor_matcher(float ratio = 0.5f) : ratio_(ratio)
    {
    }

    /// \brief setup ratio threshold for SSD filtering
    void set_ratio(float r)
    {
        ratio_ = r;
    }

    protected:
    /// \see cv::DescriptorMatcher::knnMatchImpl
    virtual void knnMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, int k,
                              cv::InputArrayOfArrays masks = cv::noArray(), bool compactResult = false) override;

    /// \see cv::DescriptorMatcher::radiusMatchImpl
    virtual void radiusMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, float maxDistance,
                                 cv::InputArrayOfArrays masks = cv::noArray(), bool compactResult = false) override;

    /// \see cv::DescriptorMatcher::isMaskSupported
    virtual bool isMaskSupported() const override
    {
        return false;
    }

    /// \see cv::DescriptorMatcher::isMaskSupported
    virtual cv::Ptr<cv::DescriptorMatcher> clone(bool emptyTrainData = false) const override
    {
        cv::Ptr<cv::DescriptorMatcher> copy = new descriptor_matcher(*this);
        if (emptyTrainData)
        {
            copy->clear();
        }
        return copy;
    }

    private:
    float ratio_ = 0.5f;
    float max_distance_ = 100.0f;
};

/// \brief Stitcher for merging images into big one
class Stitcher
{
    /// \todo design and implement
    public:
    Stitcher() : count_(0)
    {
        setParams(16, 26, 0.7f);
    }

    /// \brief Add new image
    bool addImage(cv::Mat new_image);

    /// \brief Get result image
    cv::Mat getResult();

    /// \brief Cancel last stitching
    void cancelLast();

    /// \brief Set up parameters
    void setParams(int detector_threshold, int max_match_distance, float matcher_ratio);

    /// \brief Get debug image
    cv::Mat getDebugImage(cv::Mat new_image);

    private:
    int count_;

    cv::Mat result_image_;
    std::vector<cv::KeyPoint> corners_;
    cv::Mat descriptors_;

    cv::Mat old_result_image_;
    std::vector<cv::KeyPoint> old_corners_;
    cv::Mat old_descriptors_;

    cvlib::corner_detector_fast detector_;
    cvlib::descriptor_matcher matcher_;

    int max_match_distance_ = 26;

    bool addNewImage(cv::Mat new_image);
    void savePreviousResult();
    void updateCorners(cv::Point2f offset, cv::Matx33f transform_mat, const std::vector<cv::KeyPoint>& new_corners, cv::Mat new_descriptors,
                       const std::vector<std::vector<cv::DMatch>>& pairs);
};
} // namespace cvlib

#endif // __CVLIB_HPP__
