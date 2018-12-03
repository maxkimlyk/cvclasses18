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
    const auto demo_wnd = "demo";

    cv::namedWindow(demo_wnd);

    cv::Mat ref_im = cv::imread("01.jpg");
    cv::Mat new_im = cv::imread("02.jpg");

    cvlib::Stitcher stitcher;
    stitcher.add(ref_im);
    stitcher.add(new_im);

    imshow(demo_wnd, stitcher.getAccumulatedImage());

    return 0;
}
