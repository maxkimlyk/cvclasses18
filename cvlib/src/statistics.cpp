/* CVLib utils statistics class.
 * @file
 * @date 2018-10-07
 * @author Max Kimlyk
 */

#include "utils.hpp";

namespace cvlib_utils
{

void statistics::at_start()
{
    last_update_time_ = clock::now();
    start_time_ = last_update_time_;
    frame_cnt_ = 0;
}

void statistics::at_frame_end()
{
    time now = clock::now();

    auto elapsed = std::chrono::duration_cast<duration_seconds>(now - last_update_time_);
    float as_float = elapsed.count();
    if (as_float >= 1.0f)
    {
        fps_ = frame_cnt_ / as_float;
        frame_duration_ = fps_ != 0.0 ? 1.0f / fps_ : std::numeric_limits<float>::infinity();
        last_update_time_ = now;
        frame_cnt_ = 0;
    }

    run_time_ = std::chrono::duration_cast<duration_seconds>(now - start_time_);
    ++frame_cnt_;
}

void statistics::draw(cv::Mat &frame)
{
    if (frame.channels() == 1)
        cv::cvtColor(frame, frame, cv::COLOR_GRAY2BGR);

    std::vector<std::stringstream> ss(3);

    ss[0] << "fps: " << std::setprecision(2) << std::fixed << fps_;
    ss[1] << "frame dur: " << std::setprecision(2) << std::fixed << frame_duration_ * 1000.0f << " ms";
    ss[2] << "time: " << std::setprecision(2) << std::fixed << run_time_.count() << " sec";

    const int x = 0;
    const int y = 10;
    const int width = 15;
    const double scale = 0.5;

    for (size_t i = 0; i < ss.size(); ++i)
        putText(frame, ss[i].str(), cv::Point(x, y + i * width), cv::FONT_HERSHEY_SIMPLEX, scale, cvScalar(0, 255, 0), 1);
}

}
