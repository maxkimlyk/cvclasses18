/* Descriptor matcher algorithm implementation.
 * @file
 * @date 2018-11-25
 * @author Anonymous
 */

#include "cvlib.hpp"
#include <iostream>

namespace
{
struct match
{
    int idx;
    int dist;

    void operator=(const match &other)
    {
        idx = other.idx;
        dist = other.dist;
    }
};
} // namespace

namespace cvlib
{
static int distance(int32_t x[4], int32_t y[4])
{
    int cnt = 0;
    for (int i = 0; i < 4; ++i)
    {
        int d = x[i] ^ y[i];
        for (int k = 0; k < 32; ++k)
        {
            cnt += d & 1;
            d = d >> 1;
        }
    }
    return cnt;
}

void descriptor_matcher::knnMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, int k /*unhandled*/,
                                      cv::InputArrayOfArrays masks /*unhandled*/, bool compactResult /*unhandled*/)
{
    if (trainDescCollection.empty())
        return;

    auto q_desc = queryDescriptors.getMat();
    auto& t_desc = trainDescCollection[0];

    matches.resize(q_desc.rows);

    const int init_match_distance = 999999;

    for (int i = 0; i < q_desc.rows; ++i)
    {
        ::match first_match = {0, init_match_distance};
        ::match second_match = {0, init_match_distance};

        for (int j = 0; j < t_desc.rows; ++j)
        {
            int32_t x[4] = {q_desc.at<int32_t>(i, 0), q_desc.at<int32_t>(i, 1), q_desc.at<int32_t>(i, 2), q_desc.at<int32_t>(i, 3)};
            int32_t y[4] = {t_desc.at<int32_t>(j, 0), t_desc.at<int32_t>(j, 1), t_desc.at<int32_t>(j, 2), t_desc.at<int32_t>(j, 3)};
            int dist = distance(x, y);

            if (dist > max_distance_)
                continue;

            if (dist < first_match.dist)
            {
                second_match = first_match;
                first_match.dist = dist;
                first_match.idx = j;
            }
            else if (dist < second_match.dist)
            {
                second_match.dist = dist;
                second_match.idx = j;
            }
        }

        if (static_cast<float>(first_match.dist) / static_cast<float>(second_match.dist) < ratio_)
            matches[i].emplace_back(i, first_match.idx, FLT_MAX);
    }
}

void descriptor_matcher::radiusMatchImpl(cv::InputArray queryDescriptors, std::vector<std::vector<cv::DMatch>>& matches, float maxDistance,
                                         cv::InputArrayOfArrays masks /*unhandled*/, bool compactResult /*unhandled*/)
{
    max_distance_ = maxDistance;
    knnMatchImpl(queryDescriptors, matches, 1, masks, compactResult);
}
} // namespace cvlib
