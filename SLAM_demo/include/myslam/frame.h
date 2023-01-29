//
// Created by coolas on 23-1-26.
//

#ifndef MYSLAM_FRAME_H
#define MYSLAM_FRAME_H

#include "myslam/camera.h"
#include "myslam/common_include.h"

namespace myslam{

// forward declare
struct Feature;

/**
* 帧 frame
*
* 每一帧包含：
*      1. 帧 id
*      2. 是否为关键帧，并分配关键帧ID
*      3. 位姿 pose
*      4. 特征点
* */

struct Frame{
public:
//    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frame> Ptr;

    unsigned long id_ = 0;
    unsigned long keyframe_id_ = 0;
    bool is_keyframe_ = false;        // is keyframe ?
    double time_stamp_;               // time stamps

    Sophus::SE3d pose_;               // 位姿 Tcw
    std::mutex pose_mutex_;           // pose 数据锁
    cv::Mat left_img_,right_img_;     //stereo images

    // 左目中提取的特征
    std::vector<std::shared_ptr<Feature>> features_left_;
    // 右目中提取的特征  corresponding features in right image, set to nullptr if no corresponding
    std::vector<std::shared_ptr<Feature>> features_right_;

public: // data members
    Frame(){}
    Frame(long id,double time_stamp,const Sophus::SE3d &pose,const cv::Mat &left,const cv::Mat &right);

    // set and get pose ,thread safe
    Sophus::SE3d GetPose(){
        std::unique_lock<std::mutex> lck(pose_mutex_);
        return pose_;
    }

    void SetPose(const Sophus::SE3d &pose){
        std::unique_lock<std::mutex> lck(pose_mutex_);
        pose_ = pose;
    }

    // 设置关键帧并分配关键帧id
    void SetKeyFrame();

    /// 工厂构建模式，分配id
    static std::shared_ptr<Frame> CreateFrame();

}; //namespace myslam

}
#endif //MYSLAM_FRAME_H
