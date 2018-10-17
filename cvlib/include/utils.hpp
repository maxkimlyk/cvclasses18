/* CVLib utils functions.
 * @file
 * @date 2018-09-10
 * @author Max Kimlyk
 */

#ifndef __UTILS_HPP__
#define __UTILS_HPP__

#include <chrono>

#include <opencv2/opencv.hpp>

namespace cvlib_utils
{
template<class IntegralType>
IntegralType sum_in_rect_by_integral(const cv::Mat& integral, const cv::Rect& rect)
{
    cv::Point a(rect.x, rect.y);
    cv::Point b(rect.x + rect.width, rect.y);
    cv::Point c(rect.x + rect.width, rect.y + rect.height);
    cv::Point d(rect.x, rect.y + rect.height);

    return integral.at<IntegralType>(c) - integral.at<IntegralType>(b) - integral.at<IntegralType>(d) + integral.at<IntegralType>(a);
}

void mean_stddev_by_integral(const cv::Mat& integral, const cv::Mat& integral_sq, const cv::Rect& rect, double& mean, double& std);
void mean_disp_by_integral(const cv::Mat& integral, const cv::Mat& integral_sq, const cv::Rect& rect, double& mean, double& disp);

template<class T>
int round_to_nearest_odd(T x)
{
    return 2 * (int)(x / T(2)) + 1;
}

class statistics
{
public:
    void at_start();
    void at_frame_end();
    void draw(cv::Mat &frame);

private:
    typedef std::chrono::high_resolution_clock clock;
    typedef std::chrono::time_point<clock> time;
    typedef std::chrono::duration<float> duration_seconds;

    uint64_t frame_cnt_ = 0;
    time start_time_;
    time last_update_time_;

    float fps_;
    float frame_duration_;
    duration_seconds run_time_;
};

} // namespace cvlib_utils

#endif
