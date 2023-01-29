//
// Created by coolas on 23-1-27.
//

#include "myslam/map.h"
#include "myslam/feature.h"


namespace myslam{
    /// add keyframe
    void Map::InsertKeyFrame(Frame::Ptr frame){
        current_frame_ = frame;
        // 判断是否已经插入
        if(keyframes_.find(frame->keyframe_id_) == keyframes_.end()){
            // 没有插入过,插入关键帧的时候需要知道该关键帧id
            keyframes_.insert(std::make_pair(frame->keyframe_id_,frame));
            active_keyframes_.insert(make_pair(frame->keyframe_id_, frame));
        }else{
            keyframes_[frame->keyframe_id_] = frame;
            active_keyframes_[frame->keyframe_id_] = frame;
        }
        // 控制局部地图规模，若过大需要舍弃旧的关键帧
        if (active_keyframes_.size() > num_active_keyframes_) {
            RemoveOldKeyframe();
        }
    }

    /// add mappoint
    void Map::InsertMapPoint(MapPoint::Ptr map_point){
        if (landmarks_.find(map_point->id_) == landmarks_.end()) {
            landmarks_.insert(make_pair(map_point->id_, map_point));
            active_landmarks_.insert(make_pair(map_point->id_, map_point));
        } else {
            landmarks_[map_point->id_] = map_point;
            active_landmarks_[map_point->id_] = map_point;
        }
    }

    // 将旧的关键帧设置为不活跃状态
    void Map::RemoveOldKeyframe(){
        if(current_frame_ == nullptr) return;
        // 寻找与当前帧最近与最远的两个关键帧
        double max_dis = 0,min_dis = 9999;
        double max_kf_id = 0,min_kf_id = 0;

        auto Twc = current_frame_->GetPose().inverse();
        for(auto& kf:active_keyframes_) {
            if (kf.second == current_frame_) continue;
            auto dis = (kf.second->GetPose() * Twc).log().norm(); // 误差计算公式中用到，衡量位姿之间的距离
            if (dis > max_dis) {
                max_dis = dis;
                max_kf_id = kf.first;
            }
            if (dis < min_dis) {
                min_dis = dis;
                min_kf_id = kf.first;
            }
        }
            const double min_dis_th = 0.2; //最近阈值
            Frame::Ptr frame_to_remove = nullptr;
            if(min_dis < min_dis_th){
                // 如果存在很近的帧,优先删掉最近的
                frame_to_remove = keyframes_.at(min_kf_id);
            }else{
                // 删掉最远的
                frame_to_remove = keyframes_.at(max_kf_id);
            }

            LOG(INFO) << "remove keyframe " << frame_to_remove->keyframe_id_;
            // remove keyframe and landmark observation
            active_keyframes_.erase(frame_to_remove->keyframe_id_);
            for(auto feat:frame_to_remove->features_left_){
                auto mp = feat->map_point_.lock();
                if(mp){
                    mp->RemoveObservation(feat);
                }
            }
            CleanMap();
    }
// 去掉 孤立的地图点
void Map::CleanMap() {
    int cnt_landmark_removed = 0;
    for(auto iter = active_landmarks_.begin();iter!= active_landmarks_.end();){
        if (iter->second->observed_times_ == 0) {
            iter = active_landmarks_.erase(iter);
            cnt_landmark_removed++;
        } else {
            ++iter;
        }
    }
    LOG(INFO) << "Removed " << cnt_landmark_removed << " active landmarks";
}
}

