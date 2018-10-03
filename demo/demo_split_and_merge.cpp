/* Demo application for Computer Vision Library.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <opencv2/opencv.hpp>

#include <cvlib.hpp>

int demo_split_and_merge(int argc, char* argv[])
{
    cv::VideoCapture cap(0);
    if (!cap.isOpened())
        return -1;

    cv::Mat frame;

    const auto origin_wnd = "origin";
    const auto demo_wnd = "demo";

    int stddev = 50;
    int min_chunk_size = 10;

    cv::namedWindow(demo_wnd, 1);

    cv::createTrackbar("stdev", demo_wnd, &stddev, 90);
    cv::createTrackbar("MCS", demo_wnd, &min_chunk_size, 20);

    while (cv::waitKey(30) != 27) // ESC
    {
        cap >> frame;

        int used_min_chunk_size = min_chunk_size < 1 ? 1 : min_chunk_size;

        cv::imshow(origin_wnd, frame);
        cv::imshow(demo_wnd, cvlib::split_and_merge(frame, stddev, used_min_chunk_size));
    }

    cv::destroyWindow(origin_wnd);
    cv::destroyWindow(demo_wnd);

    return 0;
}
