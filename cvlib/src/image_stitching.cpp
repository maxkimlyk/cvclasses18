/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-11-25
 * @author Anonymous
 */

#include <algorithm>
#include <cvlib.hpp>

using namespace cvlib;

namespace
{
void merge_images(cv::Mat src, cv::Mat dst)
{
    for (int i = 0; i < dst.rows; ++i)
        for (int j = 0; j < dst.cols; ++j)
        {
            cv::Vec3b src_val = src.at<cv::Vec3b>(i, j);
            if (src_val != cv::Vec3b(0, 0, 0))
                dst.at<cv::Vec3b>(i, j) = src_val;
        }
}

inline cv::Point2f transform_point(cv::Matx33f mat, cv::Point2f point)
{
    cv::Point3f tmp = mat * cv::Point3f(point.x, point.y, 1);
    return cv::Point2f(tmp.x / tmp.z, tmp.y / tmp.z);
}

void get_expanded_size_and_offset(cv::Mat new_image, cv::Matx33f homo_mat, cv::Mat old_image, cv::Size& result_size, cv::Point2f& result_offset)
{
    cv::Point2f corners[] = {cv::Point2f(0, 0),
                             cv::Point2f(0, (float)(old_image.rows)),
                             cv::Point2f((float)(old_image.cols), (float)(old_image.rows)),
                             cv::Point2f((float)(old_image.cols), 0),
                             transform_point(homo_mat, cv::Point2f(0, 0)),
                             transform_point(homo_mat, cv::Point2f(0, (float)(new_image.rows))),
                             transform_point(homo_mat, cv::Point2f((float)(new_image.cols), (float)(new_image.rows))),
                             transform_point(homo_mat, cv::Point2f((float)(new_image.cols), 0))};

    cv::Point2f* min_x = std::min_element(&corners[0], &corners[8], [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });
    cv::Point2f* min_y = std::min_element(&corners[0], &corners[8], [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });
    cv::Point2f* max_x = std::max_element(&corners[0], &corners[8], [](const cv::Point2f& a, const cv::Point2f& b) { return a.x < b.x; });
    cv::Point2f* max_y = std::max_element(&corners[0], &corners[8], [](const cv::Point2f& a, const cv::Point2f& b) { return a.y < b.y; });

    result_size = cv::Size((int)(std::ceil(max_x->x - min_x->x)), (int)(std::ceil(max_y->y - min_y->y)));
    result_offset = cv::Point2f(-min_x->x, -min_y->y);
}

cv::Matx33f translate_matrix(cv::Point2f vector)
{
    cv::Matx33f mat = cv::Matx33f::eye();
    mat.val[2] = vector.x;
    mat.val[5] = vector.y;
    return mat;
}
} // namespace

bool Stitcher::addImage(cv::Mat new_image)
{
    if (count_++ != 0)
    {
        savePreviousResult();
        return addNewImage(new_image);
    }
    else
    {
        new_image.copyTo(result_image_);
        detector_.detectAndCompute(new_image, cv::Mat(), corners_, descriptors_);
        return true;
    }
}

bool Stitcher::addNewImage(cv::Mat new_image)
{
    std::vector<cv::KeyPoint> new_corners;
    cv::Mat new_descriptors;
    std::vector<std::vector<cv::DMatch>> pairs;

    detector_.detectAndCompute(new_image, cv::Mat(), new_corners, new_descriptors);
    matcher_.radiusMatch(new_descriptors, descriptors_, pairs, (float)(max_match_distance_ + 0.5f));

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
            cv::KeyPoint& old_corner = corners_[dmatch.trainIdx];
            src_points.push_back(new_corner.pt);
            dst_points.push_back(old_corner.pt);
        }
    }

    if (src_points.size() < 4)
    {
        std::cerr << "ERROR: Not enough matches\n";
        return false;
    }

    savePreviousResult();

    cv::Matx33f homo_mat = cv::findHomography(src_points, dst_points, CV_RANSAC, 3);

    cv::Size expanded_size;
    cv::Point2f offset;
    get_expanded_size_and_offset(new_image, homo_mat, result_image_, expanded_size, offset);

    cv::Matx33f translate_mat = translate_matrix(offset);
    cv::warpPerspective(result_image_, result_image_, translate_mat, expanded_size);

    cv::Mat new_rectified;
    cv::Matx33f new_image_transform_mat = homo_mat * translate_mat;
    cv::warpPerspective(new_image, new_rectified, new_image_transform_mat, expanded_size);

    merge_images(new_rectified, result_image_);
    updateCorners(offset, new_image_transform_mat, new_corners, new_descriptors, pairs);

    return true;
}

void Stitcher::updateCorners(cv::Point2f offset, cv::Matx33f transform_mat, const std::vector<cv::KeyPoint>& new_corners, cv::Mat new_descriptors,
                   const std::vector<std::vector<cv::DMatch>>& pairs)
{
    // update old corners
    for (size_t i = 0; i < corners_.size(); ++i)
    {
        corners_[i].pt.x += offset.x;
        corners_[i].pt.y += offset.y;
    }

    // add new corners
    for (int i = 0; i < pairs.size(); ++i)
    {
        if (pairs[i].size() == 0) // if point isn't matched
        {
            cv::Point2i new_coord = transform_point(transform_mat, new_corners[i].pt);
            cv::KeyPoint new_one = new_corners[i];
            new_one.pt = new_coord;
            corners_.emplace_back(new_one);
            descriptors_.push_back(new_descriptors.row(i));
        }
    }
}

cv::Mat Stitcher::getResult()
{
    return result_image_;
}

void Stitcher::savePreviousResult()
{
    result_image_.copyTo(old_result_image_);
    old_corners_.resize(corners_.size());
    std::copy(corners_.begin(), corners_.end(), old_corners_.begin());
    descriptors_.copyTo(old_descriptors_);
}

void Stitcher::cancelLast()
{
    old_result_image_.copyTo(result_image_);
    corners_.resize(old_corners_.size());
    std::copy(old_corners_.begin(), old_corners_.end(), corners_.begin());
    old_descriptors_.copyTo(descriptors_);
}

void Stitcher::setParams(int detector_threshold, int max_match_distance, float matcher_ratio)
{
    detector_.setBrightnessTreshold(detector_threshold);
    matcher_.set_ratio(matcher_ratio);
    max_match_distance_ = max_match_distance;
}

cv::Mat Stitcher::getDebugImage(cv::Mat new_image)
{
    std::vector<cv::KeyPoint> new_corners;
    cv::Mat new_descriptors;
    std::vector<std::vector<cv::DMatch>> pairs;
    cv::Mat result;

    if (count_ < 1)
        return cv::Mat();

    detector_.detectAndCompute(new_image, cv::Mat(), new_corners, new_descriptors);
    matcher_.radiusMatch(new_descriptors, descriptors_, pairs, (float)(max_match_distance_ + 0.5f));
    cv::drawMatches(new_image, new_corners, result_image_, corners_, pairs, result);

    return result;
}
