/* Split and merge segmentation algorithm implementation.
 * @file
 * @date 2018-09-18
 * @author Anonymous
 */

#include "cvlib.hpp"
#include "utils.hpp"

namespace
{
struct descriptor : public std::vector<double>
{
    using std::vector<double>::vector;
    descriptor operator-(const descriptor& right) const
    {
        descriptor temp = *this;
        for (int i = 0; i < temp.size(); ++i)
            temp[i] -= right[i];
        return temp;
    }

    double norm_l1() const
    {
        double res = 0.0;
        for (const auto v : *this)
        {
            res += std::abs(v);
        }
        return res;
    }

    double norm_l2() const
    {
        double res = 0.0;
        for (const auto v : *this)
        {
            res += v * v;
        }
        return std::sqrt(res);
    }
};

std::vector<cv::Mat> make_filter_set(int kernel_size)
{
    const double th = CV_PI / 4;
    const double lm = 10.0;
    const double gm = 0.5;

    cv::Size size(kernel_size, kernel_size);
    std::vector<cv::Mat> filter_set;

    for (auto sig = 5; sig <= 15; sig += 5)
    {
        filter_set.emplace_back(cv::getGaborKernel(size, sig, th, lm, gm));
        filter_set.back().convertTo(filter_set.back(), CV_32F);
    }

    return filter_set;
}

void precount(const cv::Mat& image, const std::vector<cv::Mat>& filter_set, std::vector<cv::Mat>& sums, std::vector<cv::Mat>& sums_sq)
{
    size_t num_filters = filter_set.size();
    sums.resize(num_filters);
    sums_sq.resize(num_filters);

    cv::Mat response;
    for (size_t i = 0; i < num_filters; ++i)
    {
        cv::filter2D(image, response, CV_32F, filter_set[i]);
        cv::integral(response, sums[i], sums_sq[i]);
    }
}

void calculateDescriptor(const cv::Rect& rect, const std::vector<cv::Mat>& sums, const std::vector<cv::Mat>& sums_sq, descriptor& descr)
{
    descr.clear();

    size_t num_filters = sums.size();

    for (size_t i = 0; i < num_filters; ++i)
    {
        double sum = cvlib_utils::sum_in_rect_by_integral<double>(sums[i], rect);
        double sum_sq = cvlib_utils::sum_in_rect_by_integral<double>(sums_sq[i], rect);
        double inv_area = 1.0 / rect.area();
        double mean = sum * inv_area;
        double dev = std::sqrt(sum_sq * inv_area - mean * mean);
        descr.emplace_back(mean);
        descr.emplace_back(dev);
    }
}
} // namespace

namespace cvlib
{
cv::Mat select_texture(const cv::Mat& image, const cv::Rect& roi, double eps)
{
    const int kernel_size = cvlib_utils::round_to_nearest_odd(std::min(roi.height, roi.width) / 2);
    std::vector<cv::Mat> filter_set = make_filter_set(kernel_size);

    std::vector<cv::Mat> sums;
    std::vector<cv::Mat> sums_sq;
    precount(image, filter_set, sums, sums_sq);

    descriptor reference;
    calculateDescriptor(roi, sums, sums_sq, reference);

    cv::Mat res = cv::Mat::zeros(image.size(), CV_8UC1);

    descriptor test(reference.size());
    cv::Rect baseROI = roi - roi.tl();

    int end_i = image.size().width - baseROI.width;
    int end_j = image.size().height - baseROI.height;

    for (int i = 0; i <= end_i; ++i)
    {
        for (int j = 0; j <= end_j; ++j)
        {
            auto curROI = baseROI + cv::Point(i, j);
            calculateDescriptor(curROI, sums, sums_sq, test);
            descriptor diff = test - reference;
            if (diff.norm_l2() <= eps)
                res(curROI) = 255;
        }
    }

    return res;
}
} // namespace cvlib
