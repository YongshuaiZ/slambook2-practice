//
// Created by coolas on 23-1-27.
//

#ifndef MYSLAM_MAPPOINT_H
#define MYSLAM_MAPPOINT_H

#include "myslam/common_include.h"

namespace myslam{
    struct Frame;
    struct Feature;

/**
 * 3D 地图点： MapPoint
 *    1. ID
 *    2. 三维 位置
 *
* */
struct MapPoint{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<MapPoint> Ptr;
    unsigned long id_ = 0; //ID
    bool is_outlier_ = false;
    Vec3 pos_ = Vec3::Zero();      // position in world

    std::mutex data_mutex_;
    int observed_times_ = 0; // 被观测次数
    std::list<std::weak_ptr<Feature>> observations_; // 与哪些2D特征点相关联，即哪些特帧点可以观测到该地图点

public:
    MapPoint(){}
    MapPoint(long id,Vec3 position):id_(id),pos_(position){};

    Vec3 GetPos(){
        std::unique_lock<std::mutex> lck(data_mutex_);
        return pos_;
    }

    void SetPos(const Vec3 &pos){
        std::unique_lock<std::mutex> lck(data_mutex_);
        pos_ = pos;
    }

    // 添加观测次数
    void AddObservation(std::shared_ptr<Feature> feature){
        std::unique_lock<std::mutex> lck(data_mutex_);
        observations_.push_back(feature);
        observed_times_++;
    }

    void RemoveObservation(std::shared_ptr<Feature> feature);

    std::list<std::weak_ptr<Feature>> GetObs(){
        std::unique_lock<std::mutex> lck(data_mutex_);
        return observations_;
    }

    // factory function
    static MapPoint::Ptr CreateNewMappoint();
};
}
#endif //MYSLAM_MAPPOINT_H
