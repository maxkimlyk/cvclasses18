/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

void on_britness_threshold_change(int value, void* ptr)
{
    auto& detector = *(cv::Ptr<cvlib::corner_detector_fast>*)(ptr);
    detector->setBrightnessTreshold((size_t)(value));
}

void on_point_threshold_change(int value, void* ptr)
{
    auto& detector = *(cv::Ptr<cvlib::corner_detector_fast>*)(ptr);
    detector->setSuccededPointsThreshold((size_t)(value));
}

void put_points_text(cv::Mat image, size_t point_amount)
{
    const auto fontScale = 0.5;
    const auto thickness = 1;
    const auto color = cv::Scalar (0, 255, 0);

    std::stringstream ss;
    ss << "points: " << point_amount;

    cv::putText(image, ss.str(), cv::Point {2, 35}, CV_FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, 8, false);
}

int demo_corner_detector(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    const auto demo_wnd = "demo";

    cv::namedWindow(demo_wnd);

    auto detector = cvlib::corner_detector_fast::create();

    int brightness_threshold = 40;
    cv::createTrackbar("Br.Th.", demo_wnd, &brightness_threshold, 255, on_britness_threshold_change, (void*)(&detector));

    int points_threshold = 12;
    cv::createTrackbar("N", demo_wnd, &points_threshold, 16, on_point_threshold_change, (void*)(&detector));

    cv::Mat frame;
    cv::Mat frame_gray;
    std::vector<cv::KeyPoint> corners;

    utils::fps_counter fps;
    while (cv::waitKey(30) != 27) // ESC
    {
        cap >> frame;
        cv::cvtColor(frame, frame_gray, CV_BGR2GRAY);

        detector->detect(frame_gray, corners);
        cv::drawKeypoints(frame, corners, frame, cv::Scalar(0, 0, 255));
        utils::put_fps_text(frame, fps);
        put_points_text(frame, corners.size());
        cv::imshow(demo_wnd, frame);
    }

    cv::destroyWindow(demo_wnd);

    return 0;
}
