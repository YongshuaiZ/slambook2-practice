//
// Created by coolas on 23-1-27.
//

#ifndef MYSLAM_FEATURE_H
#define MYSLAM_FEATURE_H

#include <memory>
#include <opencv2/features2d.hpp>
#include "myslam/common_include.h"

namespace myslam{
/**
* 2D特征点 feature
 * 包含：
 *     1. 2D特征位置
*      2. 三角化后会被关联一个地图点
* */
struct Frame;
struct MapPoint;

struct Feature{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Feature> Ptr;

    cv::KeyPoint position_;               // 2D提取位置
    std::weak_ptr<Frame> frame_;          // 持有该feature的frame
    std::weak_ptr<MapPoint> map_point_;   // 关联地图点

    bool is_outlier_ = false;             // 是否为异常点
    bool is_on_left_image_ = true;        // 标识是否提在左图，false为右图

public:
    Feature(){};
    Feature(std::shared_ptr<Frame> frame, const cv::KeyPoint &kp)
            : frame_(frame),position_(kp){};

};

}
#endif //MYSLAM_FEATURE_H
