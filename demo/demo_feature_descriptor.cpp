/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <cvlib.hpp>
#include <opencv2/opencv.hpp>

#include "utils.hpp"

void put_points_amount_text(cv::Mat image, size_t point_amount)
{
    const auto fontScale = 0.5;
    const auto thickness = 1;
    const auto color = cv::Scalar (0, 255, 0);

    std::stringstream ss;
    ss << "points: " << point_amount;

    cv::putText(image, ss.str(), cv::Point {2, 35}, CV_FONT_HERSHEY_SIMPLEX, fontScale, color, thickness, 8, false);
}

void on_threshold_change(int value, void* ptr)
{
    auto& detector = *(cv::Ptr<cvlib::corner_detector_fast>*)(ptr);
    detector->setBrightnessTreshold((size_t)(value));
}

int demo_feature_descriptor(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    const auto demo_wnd = "demo";
    cv::namedWindow(demo_wnd);

    cv::Mat frame;
    auto detector_a = cvlib::corner_detector_fast::create();
    auto detector_b = cv::BRISK::create();
    std::vector<cv::KeyPoint> corners;
    cv::Mat descriptors;

    int brightness_threshold = 40;
    cv::createTrackbar("Det.th.", demo_wnd, &brightness_threshold, 255, on_threshold_change, (void*)(&detector_a));

    utils::fps_counter fps;
    int pressed_key = 0;
    while (pressed_key != 27) // ESC
    {
        cap >> frame;

        detector_a->detect(frame, corners);
        cv::drawKeypoints(frame, corners, frame, cv::Scalar(0, 0, 255));

        utils::put_fps_text(frame, fps);
        put_points_amount_text(frame, corners.size());
        cv::imshow(demo_wnd, frame);

        pressed_key = cv::waitKey(30);
        if (pressed_key == ' ') // space
        {
            cv::FileStorage file("descriptor.json", cv::FileStorage::WRITE | cv::FileStorage::FORMAT_JSON);

            detector_a->compute(frame, corners, descriptors);
            file << detector_a->getDefaultName() << descriptors;

            detector_b->compute(frame, corners, descriptors);
            file << "BRISK" << descriptors;

            std::cout << "Dump descriptors complete! \n";
        }

        std::cout << "Feature points: " << corners.size() << "\r";
    }

    cv::destroyWindow(demo_wnd);

    return 0;
}
