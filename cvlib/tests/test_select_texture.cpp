/* Select texture segmentation algorithm testing.
 * @file
 * @date 2018-09-30
 * @author Max Kimlyk
 */

#include <catch2/catch.hpp>
#include <iostream>

#include "cvlib.hpp"

using namespace cvlib;

// clang-format off

TEST_CASE("Select Texture: Uniform image", "[select_texture]")
{
    const cv::Mat image(5, 5, CV_8UC1, cv::Scalar{10});
    const cv::Mat ref(5, 5, CV_8UC1, cv::Scalar{255});
    const auto res = select_texture(image, cv::Rect {1, 1, 2, 2}, 5.0);
    std::cout << res << std::endl;
    REQUIRE(0 == cv::countNonZero(ref - res));
}

TEST_CASE("Select Texture: Texture image", "[select_texture]")
{
    const cv::Mat image = (cv::Mat_<uint8_t>(11, 11) <<
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1,19, 1, 1, 1, 1, 1, 1, 1, 1,
        1,19,19,19, 1, 1, 1, 1, 1, 1, 1,
        1, 1,19, 1,19, 1, 1, 1, 1, 1, 1,
        1, 1, 1,19, 1, 1,19, 1, 1, 1, 1,
        1, 1, 1, 1, 1,19,19,19, 1, 1, 1,
        1, 1, 1, 1, 1, 1,19, 1,19, 1, 1,
        1, 1, 1, 1, 1, 1, 1,19, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
        1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1
    );

    const cv::Mat ref = (cv::Mat_<uint8_t>(11, 11) <<
        255, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,
        255, 255, 255, 255, 255,   0,   0,   0,   0,   0,   0,
        255, 255, 255, 255, 255, 255, 255,   0,   0,   0,   0,
        255, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,
        255, 255, 255, 255, 255, 255, 255, 255, 255,   0,   0,
        0,   0,   0,   255, 255, 255, 255, 255, 255,   0,   0,
        0,   0,   0,   255, 255, 255, 255, 255, 255,   0,   0,
        0,   0,   0,   255, 255, 255, 255, 255, 255,   0,   0,
        0,   0,   0,     0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,     0,   0,   0,   0,   0,   0,   0,   0,
        0,   0,   0,     0,   0,   0,   0,   0,   0,   0,   0
    );

    const auto res = select_texture(image, cv::Rect {1, 1, 4, 4}, 5.0);
    REQUIRE(0 == cv::countNonZero(ref - res));
}

// clang-format on
