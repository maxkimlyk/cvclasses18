/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Max Kimlyk
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

int demo_motion_segmentation(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    const auto main_wnd = "main";
    const auto demo_wnd = "demo";
    const auto background_wnd = "background";

    int threshold = 50;
    const int max_threshold = 255;
    const double threshold_multiplyer = 10.0 / max_threshold;

    int learning_rate = 50;
    const int learning_rate_max = 100;

    cv::namedWindow(main_wnd);
    cv::namedWindow(demo_wnd);
    cv::namedWindow(background_wnd);
    cv::createTrackbar("th", demo_wnd, &threshold, max_threshold);
    cv::createTrackbar("LR", demo_wnd, &learning_rate, learning_rate_max);

    cv::Mat frame;
    cv::Mat frame_gray;
    cap >> frame;
    cv::cvtColor(frame, frame_gray, CV_BGR2GRAY);
    auto mseg = cvlib::motion_segmentation(frame_gray);

    cv::Mat frame_mseg;
    cv::Mat blured;
    cv::Mat background;

    utils::fps_counter fps;
    while (cv::waitKey(30) != 27) // ESC
    {
        cap >> frame;
        cv::cvtColor(frame, frame_gray, CV_BGR2GRAY);
        cv::imshow(main_wnd, frame_gray);

        cv::GaussianBlur(frame_gray, blured, cv::Size(5, 5), 5);

        mseg.setVarThreshold(threshold * threshold_multiplyer);
        mseg.apply(blured, frame_mseg, (double)(learning_rate) / learning_rate_max * 0.01);

        if (!frame_mseg.empty())
        {
            utils::put_fps_text(frame_mseg, fps);
            cv::imshow(demo_wnd, frame_mseg);
        }

        mseg.getBackgroundImage(background);
        cv::imshow(background_wnd, background);
    }

    cv::destroyWindow(main_wnd);
    cv::destroyWindow(demo_wnd);
    cv::destroyWindow(background_wnd);

    return 0;
}
