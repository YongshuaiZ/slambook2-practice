//
// Created by coolas on 23-1-27.
//
#include "myslam/mappoint.h"
#include "myslam/feature.h"

namespace myslam{

MapPoint::Ptr MapPoint::CreateNewMappoint() {
    static long factory_id = 0;
    MapPoint::Ptr new_mappoint(new MapPoint);
    new_mappoint->id_ = factory_id++;
    return new_mappoint;
}
///  一个地图点可以被多个 2D特帧 观测到
///  此处 将特征 feature 所对应的地图点 删除
void MapPoint::RemoveObservation(std::shared_ptr<Feature> feature) {
    std::unique_lock<std::mutex> lck(data_mutex_);
    for(auto iter = observations_.begin();iter != observations_.end();iter++)
    {
        if(iter->lock() == feature){
            observations_.erase(iter);
            feature->map_point_.reset();
            observed_times_--;
            break;
        }
    }
}
}