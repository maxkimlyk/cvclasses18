/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-11-25
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

int demo_image_stitching(int argc, char* argv[])
{
    const int MAX_RATIO = 100;

    const auto control_wnd = "control";
    const auto demo_wnd = "demo";
    const auto dbg_wnd = "dbg";

    cv::namedWindow(control_wnd);
    cv::namedWindow(demo_wnd);
    cv::namedWindow(dbg_wnd);

    auto detector = cvlib::corner_detector_fast();
    auto matcher = cvlib::descriptor_matcher();

    int ratio = 70;
    int corner_threshold = 16;
    int max_distance = 26;

    cvlib::Stitcher stitcher;

    cv::createTrackbar("ratio %", control_wnd, &ratio, MAX_RATIO);
    cv::createTrackbar("thresh", control_wnd, &corner_threshold, 255);
    cv::createTrackbar("max dist", control_wnd, &max_distance, 128);

    cv::Mat old_image = cv::imread("p01.jpg");
    cv::Mat new_image = cv::imread("p02.jpg");

    cv::Mat main_frame;
    cv::Mat demo_frame;
    cv::Mat dbg_frame;
    utils::fps_counter fps;

    stitcher.addImage(old_image);

    int pressed_key = 0;
    while (pressed_key != 27) // ESC
    {
        float ratiof = static_cast<float>(ratio) / MAX_RATIO;
        stitcher.setParams(corner_threshold, max_distance, ratiof);

        dbg_frame = stitcher.getDebugImage(new_image);
        cv::imshow(dbg_wnd, dbg_frame);

        demo_frame = stitcher.getResult();
        cv::imshow(demo_wnd, demo_frame);

        pressed_key = cv::waitKey(30);
        if (pressed_key == ' ')
        {
            stitcher.addImage(new_image);
        }
        else if (pressed_key == 'z')
            stitcher.cancelLast();
    }

    cv::destroyWindow(demo_wnd);

    return 0;
}
