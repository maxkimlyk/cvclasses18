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
    const int MAX_BLUR = 100;
    const double BLUR_COEF = 5.0;

    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    const auto camera_wnd = "camera";
    const auto control_wnd = "control";
    const auto demo_wnd = "demo";
    const auto dbg_wnd = "dbg";

    cv::namedWindow(camera_wnd);
    cv::namedWindow(control_wnd);
    cv::namedWindow(demo_wnd);
    cv::namedWindow(dbg_wnd);

    auto detector = cvlib::corner_detector_fast();
    auto matcher = cvlib::descriptor_matcher();

    int ratio = 70;
    int corner_threshold = 16;
    int max_distance = 26;
    int blur_value = 20;

    cvlib::Stitcher stitcher;

    cv::createTrackbar("ratio %", control_wnd, &ratio, MAX_RATIO);
    cv::createTrackbar("thresh", control_wnd, &corner_threshold, 255);
    cv::createTrackbar("max dist", control_wnd, &max_distance, 128);
    cv::createTrackbar("blur %", control_wnd, &blur_value, MAX_BLUR);

    cv::Mat camera_frame;
    cv::Mat demo_frame;
    cv::Mat dbg_frame;
    utils::fps_counter fps;

    int pressed_key = 0;
    while (pressed_key != 27) // ESC
    {
        cap >> camera_frame;
        double blur_sigma = static_cast<double>(blur_value) / MAX_BLUR * BLUR_COEF;
        cv::GaussianBlur(camera_frame, camera_frame, cv::Size(5, 5), blur_sigma);
        cv::imshow(camera_wnd, camera_frame);

        float ratiof = static_cast<float>(ratio) / MAX_RATIO;
        stitcher.setParams(corner_threshold, max_distance, ratiof);

        dbg_frame = stitcher.getDebugImage(camera_frame);
        cv::imshow(dbg_wnd, dbg_frame);

        demo_frame = stitcher.getResult();
        if (!demo_frame.empty())
            cv::imshow(demo_wnd, demo_frame);

        pressed_key = cv::waitKey(30);
        if (pressed_key == ' ')
            stitcher.addImage(camera_frame);
        else if (pressed_key == 'z')
            stitcher.cancelLast();
    }

    cv::destroyWindow(camera_wnd);
    cv::destroyWindow(control_wnd);
    cv::destroyWindow(demo_wnd);
    cv::destroyWindow(dbg_wnd);

    return 0;
}

int demo_image_stitching_static(int argc, char* argv[])
{
    const int MAX_RATIO = 100;
    const int MAX_BLUR = 100;
    const double BLUR_COEF = 5.0;

    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    const auto control_wnd = "control";
    const auto demo_wnd = "demo";
    const auto dbg_wnd = "dbg";

    cv::namedWindow(control_wnd);
    cv::namedWindow(demo_wnd);
    cv::namedWindow(dbg_wnd);

    cv::Mat images[] = {cv::imread("p01.jpg"), cv::imread("p02.jpg"), cv::imread("p03.jpg"),
                        cv::imread("p04.jpg"), cv::imread("p05.jpg"), cv::imread("p06.jpg")};
    const int IMAGES_AMOUNT = sizeof(images) / sizeof(images[0]);

    auto detector = cvlib::corner_detector_fast();
    auto matcher = cvlib::descriptor_matcher();

    int ratio = 70;
    int corner_threshold = 16;
    int max_distance = 26;
    int blur_value = 20;

    cvlib::Stitcher stitcher;

    cv::createTrackbar("ratio %", control_wnd, &ratio, MAX_RATIO);
    cv::createTrackbar("thresh", control_wnd, &corner_threshold, 255);
    cv::createTrackbar("max dist", control_wnd, &max_distance, 128);
    cv::createTrackbar("blur %", control_wnd, &blur_value, MAX_BLUR);

    cv::Mat demo_frame;
    cv::Mat dbg_frame;
    utils::fps_counter fps;

    int count = 0;

    int pressed_key = 0;
    while (pressed_key != 27) // ESC
    {
        cv::Mat camera_frame;
        images[count].copyTo(camera_frame);

        double blur_sigma = static_cast<double>(blur_value) / MAX_BLUR * BLUR_COEF;
        cv::GaussianBlur(camera_frame, camera_frame, cv::Size(5, 5), blur_sigma);

        float ratiof = static_cast<float>(ratio) / MAX_RATIO;
        stitcher.setParams(corner_threshold, max_distance, ratiof);

        dbg_frame = stitcher.getDebugImage(camera_frame);
        cv::imshow(dbg_wnd, dbg_frame);

        demo_frame = stitcher.getResult();
        if (!demo_frame.empty())
            cv::imshow(demo_wnd, demo_frame);

        pressed_key = cv::waitKey(30);
        if (pressed_key == ' ')
        {
            stitcher.addImage(camera_frame);
            count = (count + 1) % IMAGES_AMOUNT;
        }
        else if (pressed_key == 'z')
        {
            stitcher.cancelLast();
            count = (count - 1) % IMAGES_AMOUNT;
        }
    }

    cv::destroyWindow(control_wnd);
    cv::destroyWindow(demo_wnd);
    cv::destroyWindow(dbg_wnd);

    return 0;
}
