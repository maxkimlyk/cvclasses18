/* FAST corner detector algorithm testing.
 * @file
 * @date 2018-09-05
 * @author Anonymous
 */

#include <catch2/catch.hpp>

#include "cvlib.hpp"

using namespace cvlib;

// clang-format off

TEST_CASE("simple check", "[corner_detector_fast]")
{
    auto fast = corner_detector_fast::create();
    std::vector<cv::KeyPoint> out;

    SECTION("empty image")
    {
        cv::Mat image(10, 10, CV_8UC1);
        fast->detect(image, out);
        REQUIRE(out.empty());
    }

    SECTION("uniform image")
    {
        cv::Mat image(10, 10, CV_8UC1, cv::Scalar(110));
        fast->detect(image, out);
        REQUIRE(out.empty());
    }
}

TEST_CASE("shapes", "[corner_detector_fast]")
{
    auto fast = corner_detector_fast::create();
    std::vector<cv::KeyPoint> out;

    SECTION("point")
    {
        cv::Mat image = (cv::Mat_<uint8_t>(9, 9) <<
                0,   0,   0,   0,   0,   0,   0,   0,   0,
                0,   0,   0,   0,   0,   0,   0,   0,   0,
                0,   0,   3,   2,   0,   0,   0,   0,   0,
                0,   0,   0,   4,   3,   0,   0,   0,   0,
                0,   0,   3,   0,  80,   0,   0,   0,   0,
                0,   0,   5,   0,   0,   0,   3,   0,   0,
                0,   0,   0,   0,   0,   0,   5,   0,   0,
                0,   0,   0,   0,   0,   0,   0,   0,   0,
                0,   0,   0,   0,   0,   0,   0,   0,   0);

        fast->detect(image, out);
        REQUIRE(1 == out.size());
    }

    SECTION("two points")
    {
        cv::Mat image = (cv::Mat_<uint8_t>(9, 9) <<
                0,   0,   0,   0,   0,   0,   0,   0,   0,
                0,   0,   0,   0,   0,   0,   0,   0,   0,
                0,   0,   3,   2,   0,   0,   0,   0,   0,
                0,   0,   0,   4,   3,   0,   0,   0,   0,
                0,   0,   3,  110, 80,  10,   0,   0,   0,
                0,   0,   5,   0,   0,   0,   3,   0,   0,
                0,   0,   0,   0,   0,   0,   5,   0,   0,
                0,   0,   0,   0,   0,   0,   0,   0,   0,
                0,   0,   0,   0,   0,   0,   0,   0,   0);

        fast->detect(image, out);
        REQUIRE(2 == out.size());
    }

    SECTION("strip")
    {
        cv::Mat image = (cv::Mat_<uint8_t>(9, 9) <<
                0,   0,   0, 100, 100, 100, 100, 100,   0,
                0,   0,   0, 100, 100, 100, 100, 100,   0,
                0,   0,   0, 100, 100, 100, 100, 100,   0,
                0,   0,   0, 100, 100, 100, 100, 100,   0,
                0,   0,   0, 100, 100, 100, 100, 100,   0,
                0,   0,   0, 100, 100, 100, 100, 100,   0,
                0,   0,   0, 100, 100, 100, 100, 100,   0,
                0,   0,   0, 100, 100, 100, 100, 100,   0,
                0,   0,   0, 100, 100, 100, 100, 100,   0);

        fast->detect(image, out);
        REQUIRE(0 == out.size());
    }

    SECTION("corner")
    {
        cv::Mat image = (cv::Mat_<uint8_t>(9, 9) <<
                0,   0,   0,   0,   0,   0,   0,   0,   0,
                0,   0,   0,   0,   0,   0,   0,   0,   0,
                0,   0,   0,   0,   0,   0,   0,   0,   0,
                0,   0,   0,   0,   0,   0,   0,   0,   0,
                0,   0,   0,   0,  80,   0,   0,   0,   0,
                0,   0,   0,  80,  80,  80,   0,   0,   0,
                0,   0,  00,  80,  80,  80,  80,   0,   0,
                0,  80,  80,  80,  80,  80,  80,  80,   0,
                0,  80,  80,  80,  80,  80,  80,  80,  80);

        fast->detect(image, out);
        REQUIRE(1 == out.size());
        float a = out[0].angle;
        CHECK(std::abs(2 * M_PI - out[0].angle) < 2 * M_PI * 0.05);
    }
}

TEST_CASE("threshold", "[corner_detector_fast]")
{
    auto fast = corner_detector_fast::create();
    std::vector<cv::KeyPoint> out;

    cv::Mat image = (cv::Mat_<uint8_t>(9, 9) <<
            0,   0,   0,   0,   0,   0,   0,   0,   0,
            0,   0,   0,  75,   0,   5,   0,   0,   0,
            0,   0,  70,   0,   0,   0,  10,   0,   0,
            0,  65,   0,   0,   0,   0,   0,  15,   0,
            0,  60,   0,   0,  80,   0,   0,  20,   0,
            0,  55,   0,   0,   0,   0,   0,  25,   0,
            0,   0,  50,   0,   0,   0,  30,   0,   0,
            0,   0,   0,  45,  40,  35,   0,   0,   0,
            0,   0,   0,   0,   0,   0,   0,   0,   0);

    SECTION("N")
    {

        fast->setBrightnessTreshold(40);

        for (size_t t = 16; t >= 10; --t)
        {
            fast->setSuccededPointsThreshold(t);
            fast->detect(image, out);
            CHECK(0 == out.size());
        }

        for (size_t t = 9; t >= 5; --t)
        {
            fast->setSuccededPointsThreshold(t);
            fast->detect(image, out);
            CHECK(1 == out.size());
        }
    }

    SECTION("brighness")
    {
        cv::Mat image = (cv::Mat_<uint8_t>(9, 9) <<
                0,   0,   0,   0,   0,   0,   0,   0,   0,
                0,   0,   0,  75,   0,   5,   0,   0,   0,
                0,   0,  70,   0,   0,   0,  10,   0,   0,
                0,  65,   0,   0,   0,   0,   0,  15,   0,
                0,  60,   0,   0,  80,   0,   0,  20,   0,
                0,  55,   0,   0,   0,   0,   0,  25,   0,
                0,   0,  50,   0,   0,   0,  30,   0,   0,
                0,   0,   0,  45,  40,  35,   0,   0,   0,
                0,   0,   0,   0,   0,   0,   0,   0,   0);


        fast->setSuccededPointsThreshold(12);

        fast->setBrightnessTreshold(10);
        fast->detect(image, out);
        CHECK(1 == out.size());

        fast->setBrightnessTreshold(20);
        fast->detect(image, out);
        CHECK(1 == out.size());

        fast->setBrightnessTreshold(25);
        fast->detect(image, out);
        CHECK(1 == out.size());

        fast->setBrightnessTreshold(30);
        fast->detect(image, out);
        CHECK(0 == out.size());
    }
}

// clang-format on
