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
static int distance(cv::Mat x, cv::Mat y)
{
    int cnt = 0;
    for (int i = 0; i < x.cols; ++i)
    {
        int d = x.at<int32_t>(0, i) ^ y.at<int32_t>(0, i);
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
            cv::Mat x = q_desc.row(i);
            cv::Mat y = t_desc.row(j);
            int dist = distance(x, y);

            if (dist > (int)(max_distance_))
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
