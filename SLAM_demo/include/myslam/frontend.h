//
// Created by coolas on 23-1-27.
//

#ifndef MYSLAM_FRONTEND_H
#define MYSLAM_FRONTEND_H

#include <opencv2/features2d.hpp>

#include "myslam/common_include.h"
#include "myslam/frame.h"
#include "myslam/map.h"
#include "viewer.h"

class Camera;

namespace myslam{

enum class FrontendStatus {INITING,TRACKING_GOOD,TRACKING_BAD,LOST};
/**
 * 前端
 * 估计当前帧pose,在满足关键帧条件时向地图加入关键帧并触发优化
 * */
class Frontend{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Frontend> Ptr;

    Frontend();

private:
    /**
     * Track in normal mode
     * @return true if success
     */
    bool Track();

    int TrackLastFrame();
    ///visual_odometry调用接口，添加一个帧并计算其定位结果
    bool AddFrame(Frame::Ptr frame);

    /**
     * Try init the frontend with stereo images saved in current_frame_
     * @return true if success
     * */
     bool StereoInit();

    /**
        * Detect features in left image in current_frame_
        * keypoints will be saved in current_frame_
        * @return
   */
    int DetectFeatures();

    /**
     * Find the corresponding features in right image of current_frame_
     * @return num of features found
     */
    int FindFeaturesInRight();

    /**
     * Build the initial map with single image
     * @return true if succeed
     */
    int BuildInitMap();

    int EstimateCurrentPose();

    bool InsertKeyframe();

    void SetObservationsForKeyFrame();

    int TriangulateNewPoints();

    bool Reset();

public:
    // data
    FrontendStatus status_ = FrontendStatus::INITING;

    Frame::Ptr current_frame_  = nullptr;      // 当前帧
    Frame::Ptr last_frame_ = nullptr ;         // last frame
    Camera::Ptr camera_left_ = nullptr;        // 左侧相机
    Camera::Ptr camera_right_ = nullptr;       // 右侧相机

    Map::Ptr map_ = nullptr;

    Sophus::SE3d relative_motion_;   // 当前帧与上一帧的相对运动，用于估计当前帧pose初值

    std::shared_ptr<Backend> backend_ = nullptr;
    std::shared_ptr<Viewer> viewer_ = nullptr;

    int tracking_inliers_ = 0; // inliers, used for testing new keyframes

    // params
    int num_features_ = 200;
    int num_features_init_ = 100;
    int num_features_tracking_ = 50;
    int num_features_tracking_bad_ = 20;
    int num_features_needed_for_keyframe_ = 80;

    // utilites
    cv::Ptr<cv::GFTTDetector> gftt_; // feature detector in opencv

};
}
#endif //MYSLAM_FRONTEND_H
