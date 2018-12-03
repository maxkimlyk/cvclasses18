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
    }

    void add(cv::Mat new_part)
    {
        if (count_ == 0)
        {
            new_part.copyTo(acc_image_);
            detector_.detectAndCompute(new_part, cv::Mat(), acc_corners_, acc_descriptors);
        }
        else
        {
            std::vector<cv::KeyPoint> new_corners;
            cv::Mat new_descriptors;
            detector_.detectAndCompute(new_part, cv::Mat(), new_corners, new_descriptors);

            std::vector<std::vector<cv::DMatch>> pairs;
            matcher_.radiusMatch(new_descriptors, acc_descriptors, pairs, 999.0f);

            cv::Mat dbg_frame;
            cv::drawMatches(new_part, new_corners, acc_image_, acc_corners_, pairs, dbg_frame);
            cv::namedWindow("dbg");
            cv::imshow("dbg", dbg_frame);

            std::vector<cv::Point2f> src_points;
            std::vector<cv::Point2f> dst_points;

            src_points.reserve(pairs.size());
            dst_points.reserve(pairs.size());

            for (size_t i = 0; i < pairs.size(); ++i)
            {
                for (size_t j = 0; j < pairs[i].size(); ++j)
                {
                    cv::DMatch& dmatch = pairs[i][j];
                    cv::KeyPoint& new_corner = new_corners[dmatch.queryIdx];
                    cv::KeyPoint& old_corner = acc_corners_[dmatch.trainIdx];
                    src_points.push_back(new_corner.pt);
                    dst_points.push_back(old_corner.pt); // << or vise versa?
                }
            }

            if (src_points.size() < 4)
            {
                std::cout << "ERROR: Not enough matches\n";
                return;
            }

            cv::Mat homo_mat = cv::findHomography(src_points, dst_points, CV_RANSAC, 3);

            cv::Mat new_part_rectified;
            cv::warpPerspective(new_part, new_part_rectified, homo_mat, acc_image_.size());

            cv::namedWindow("test");
            cv::imshow("test", new_part_rectified);
        }
        count_++;
    }

    cv::Mat getAccumulatedImage()
    {
        return acc_image_;
    }

    private:
    cv::Mat acc_image_;
    int count_;

    std::vector<cv::KeyPoint> acc_corners_;
    cv::Mat acc_descriptors;

    cvlib::corner_detector_fast detector_;
    cvlib::descriptor_matcher matcher_;
};
} // namespace cvlib

#endif // __CVLIB_HPP__
