//
// Created by coolas on 23-1-27.
//
#include <opencv2/opencv.hpp>

#include "myslam/algorithm.h"
#include "myslam/backend.h"
#include "myslam/config.h"
#include "myslam/feature.h"
#include "myslam/frontend.h"
#include "myslam/g2o_types.h"
#include "myslam/map.h"
#include "myslam/viewer.h"

namespace myslam{

Frontend::Frontend() {
  gftt_ = cv::GFTTDetector::create(Config::Get<int>("num_features"), 0.01, 20);
  num_features_init_ = Config::Get<int>("num_features_init");
  num_features_ = Config::Get<int>("num_features");
}

bool Frontend::AddFrame(myslam::Frame::Ptr frame){
     current_frame_ = frame;
     switch(status_){
         case FrontendStatus::INITING:
             StereoInit();
             break;
         case FrontendStatus::TRACKING_GOOD:
         case FrontendStatus::TRACKING_BAD:
             Track();
             break;
         case FrontendStatus::LOST:
             Reset();
             break;
        }
        last_frame_ = current_frame_;
        return true;
    }

bool Frontend::Track() {

    if(last_frame_){
        current_frame_->SetPose(relative_motion_ * last_frame_->GetPose()); // Tcl * Tlw
    }

    int num_track_last = TrackLastFrame();  //last frame 和current frame之间特征匹配 【光流法】
    tracking_inliers_ = EstimateCurrentPose(); // 使用g2o进行图优化，仅优化位姿

    if (tracking_inliers_ > num_features_tracking_) {
        // tracking good
        status_ = FrontendStatus::TRACKING_GOOD;
    } else if (tracking_inliers_ > num_features_tracking_bad_) {
        // tracking bad
        status_ = FrontendStatus::TRACKING_BAD;
    } else {
        // lost
        status_ = FrontendStatus::LOST;
    }
    InsertKeyframe();

    relative_motion_ = current_frame_->GetPose() * last_frame_->GetPose().inverse();

    if (viewer_) viewer_->AddCurrentFrame(current_frame_);
    return true;
}

// 插入关键帧
// 每当有一个新的关键帧，则激活后端优化
bool Frontend::InsertKeyframe() {
    if (tracking_inliers_ >= num_features_needed_for_keyframe_) {
        // still have enough features, don't insert keyframe
        return false;
    }

    // current frame is a new keyframe
    current_frame_->SetKeyFrame();
    map_->InsertKeyFrame(current_frame_);

    LOG(INFO) << "Set frame " << current_frame_->id_ << " as keyframe "
              << current_frame_->keyframe_id_;

    SetObservationsForKeyFrame();  // 对mappoint设置观测 ，看mappoint被哪些特帧观测到
    DetectFeatures();  // detect new features 如果是关键帧才能执行到这一步，是关键帧的话其跟踪到的内点数目就会相应不足，需要补充

    // track in right image
    FindFeaturesInRight();
    // triangulate map points
    TriangulateNewPoints();

    LOG(INFO) << "三角化 success";
    // update backend because we have a new keyframe
    backend_->UpdateMap();

    if (viewer_) viewer_->UpdateMap();

    return true;

}

int Frontend::TriangulateNewPoints() {
        std::vector<Sophus::SE3d> poses{camera_left_->pose(), camera_right_->pose()};
        Sophus::SE3d current_pose_Twc = current_frame_->GetPose().inverse();
        int cnt_triangulated_pts = 0;
        for (size_t i = 0; i < current_frame_->features_left_.size(); ++i) {
            if (current_frame_->features_left_[i]->map_point_.expired() &&
                current_frame_->features_right_[i] != nullptr) {
                // 左图的特征点未关联地图点且存在右图匹配点，尝试三角化
                std::vector<Vec3> points{
                        camera_left_->pixel2camera(
                                Vec2(current_frame_->features_left_[i]->position_.pt.x,
                                     current_frame_->features_left_[i]->position_.pt.y)),
                        camera_right_->pixel2camera(
                                Vec2(current_frame_->features_right_[i]->position_.pt.x,
                                     current_frame_->features_right_[i]->position_.pt.y))};
                Vec3 pworld = Vec3::Zero();

                if (triangulation(poses, points, pworld) && pworld[2] > 0) {
                    auto new_map_point = MapPoint::CreateNewMappoint();
                    pworld = current_pose_Twc * pworld;
                    new_map_point->SetPos(pworld);
                    new_map_point->AddObservation(
                            current_frame_->features_left_[i]);
                    new_map_point->AddObservation(
                            current_frame_->features_right_[i]);

                    current_frame_->features_left_[i]->map_point_ = new_map_point;
                    current_frame_->features_right_[i]->map_point_ = new_map_point;
                    map_->InsertMapPoint(new_map_point);
                    cnt_triangulated_pts++;
                }
            }
        }
        LOG(INFO) << "new landmarks: " << cnt_triangulated_pts;
        return cnt_triangulated_pts;
    }

    //查找当前帧中的特征，看是否对应已有的地图点，若对应则为地图点添加当前帧内的特征观测
void Frontend::SetObservationsForKeyFrame() {
        for (auto &feat : current_frame_->features_left_) {
            auto mp = feat->map_point_.lock();
            if (mp) mp->AddObservation(feat);
        }
    }


int Frontend::EstimateCurrentPose(){
    // setup g2o
    typedef g2o::BlockSolver_6_3 BlockSolverType;
    typedef g2o::LinearSolverDense<BlockSolverType::PoseMatrixType> LinearSolverType;
    auto solver = new g2o::OptimizationAlgorithmLevenberg(
            g2o::make_unique<BlockSolverType>(
                    g2o::make_unique<LinearSolverType>()));
    g2o::SparseOptimizer optimizer;
    optimizer.setAlgorithm(solver);

    // vertex
    VertexPose *vertex_pose = new VertexPose();  // camera vertex_pose
    vertex_pose->setId(0);
    vertex_pose->setEstimate(current_frame_->GetPose());
    optimizer.addVertex(vertex_pose);

    // K
    Mat33 K = camera_left_->K();

    // edges
    int index = 1;
    std::vector<EdgeProjectionPoseOnly *> edges;
    std::vector<Feature::Ptr> features;
    for(size_t i=0;i<current_frame_->features_left_.size();i++)
    {
        auto mp = current_frame_->features_left_[i]->map_point_.lock();
        if(mp){  //这里就涉及到前面在TrackLastFrame()函数里面提到的，有些特征虽然被跟踪到了，但是并没有受到三角化，即没有map_point_
            //这里便对feature有没有map_point_进行判断，有则可以往下进行重投影，没有则不行，因为重投影需要点的3D位置TrackLastFrame()函数里面提到的，有些特征虽然被跟踪到了，但是并没有受到三角化，即没有map_point_
            features.push_back(current_frame_->features_left_[i]);
            EdgeProjectionPoseOnly *edge =
                    new EdgeProjectionPoseOnly(mp->pos_,K);
            edge->setId(index);
            edge->setVertex(0,vertex_pose);
            edge->setMeasurement(
                    toVec2(current_frame_->features_left_[i]->position_.pt)
                    );
            edge->setInformation(Eigen::Matrix2d::Identity());
            edge->setRobustKernel(new g2o::RobustKernelHuber);
            edges.push_back(edge);
            optimizer.addEdge(edge);
            index++;
        }
    }

    // estimate the Pose the determine the outliers
    const double chi2_th = 5.991;
    int cnt_outlier = 0;
    for(int iteration = 0;iteration<4;iteration++)
    {
        vertex_pose->setEstimate(current_frame_->GetPose());
        optimizer.initializeOptimization();
        optimizer.optimize(10);
        cnt_outlier = 0;

        // count the outliers
        for(size_t i=0;i<edges.size();i++)
        {
            auto e = edges[i];
            if(features[i]->is_outlier_){
                e->computeError();
            }

            if(e->chi2() > chi2_th){
                features[i]->is_outlier_ = true;
                e->setLevel(1);
                cnt_outlier++;
            }else{
                features[i]->is_outlier_ = false;
                e->setLevel(0);
            }

            if (iteration == 2) {
                e->setRobustKernel(nullptr);
            }
        }

    }
    LOG(INFO) << "Outlier/Inlier in pose estimating: " << cnt_outlier << "/"
              << features.size() - cnt_outlier;

    // Set pose and outlier
    current_frame_->SetPose(vertex_pose->estimate());

    LOG(INFO) << "Current Pose = \n" << current_frame_->GetPose().matrix();

    for (auto &feat : features) {
        if (feat->is_outlier_) {
            feat->map_point_.reset();
            feat->is_outlier_ = false;  // maybe we can still use it in future
        }
    }
    return features.size() - cnt_outlier;

}

int Frontend::TrackLastFrame() {
        // use LK flow to estimate points in the right image
        std::vector<cv::Point2f> kps_last, kps_current;
        for (auto &kp : last_frame_->features_left_) {
            if (kp->map_point_.lock()) {
                // use project point
                auto mp = kp->map_point_.lock();
                auto px =
                        camera_left_->world2pixel(mp->pos_, current_frame_->GetPose());
                kps_last.push_back(kp->position_.pt);
                kps_current.push_back(cv::Point2f(px[0], px[1]));
            } else {
                kps_last.push_back(kp->position_.pt);
                kps_current.push_back(kp->position_.pt);
            }
        }

        std::vector<uchar> status;
        Mat error;
        cv::calcOpticalFlowPyrLK(
                last_frame_->left_img_, current_frame_->left_img_, kps_last,
                kps_current, status, error, cv::Size(11, 11), 3,
                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                                 0.01),
                cv::OPTFLOW_USE_INITIAL_FLOW);

        int num_good_pts = 0;

        for (size_t i = 0; i < status.size(); ++i) {
            if (status[i]) {
                cv::KeyPoint kp(kps_current[i], 7);
                Feature::Ptr feature(new Feature(current_frame_, kp));
                feature->map_point_ = last_frame_->features_left_[i]->map_point_;
                current_frame_->features_left_.push_back(feature);
                num_good_pts++;
            }
        }

        LOG(INFO) << "Find " << num_good_pts << " in the last image.";
        return num_good_pts;
    }
/**
 * 双目初始化 StereoInit()
 * 在初始化阶段，根据左右目之间的光流匹配，寻找可以三角化的地图点，成功时建立初始地图
 * */
bool Frontend::StereoInit() {
    int num_features_left = DetectFeatures(); /// 提取左目中特帧， 完成 current_frame_->features_left_
    int num_coor_features = FindFeaturesInRight();  /// 特征匹配 左图和右图中所相对应的特征 完成 current_frame_->features_right_
    if(num_coor_features < num_features_init_){
        return false;
    }

    // 建立初始Local Map
    bool build_map_success = BuildInitMap();
    if(build_map_success){
        status_ = FrontendStatus::TRACKING_GOOD;
        if(viewer_){
            viewer_->AddCurrentFrame(current_frame_);
            viewer_->UpdateMap();
        }
       return true;
    }
    return false;
}

// 构建新帧, 返回帧中2D特帧点的个数
int Frontend::DetectFeatures(){
    cv::Mat mask(current_frame_->left_img_.size(),CV_8UC1,255);
    // 绘图函数
    for(auto &feat:current_frame_->features_left_){
        cv::rectangle(mask, feat->position_.pt - cv::Point2f(10, 10),
                      feat->position_.pt + cv::Point2f(10, 10), 0, CV_FILLED);
    }

    std::vector<cv::KeyPoint> keypoints;
    /// detect函数，第三个参数是用来指定特征点选取区域的，一个和原图像同尺寸的掩膜，其中非0区域代表detect函数感兴趣的提取区域，相当于为
    /// detect函数明确了提取的大致位置
    gftt_->detect(current_frame_->left_img_,keypoints,mask);  //
    int cnt_detected = 0;
    for(auto &kp:keypoints)
    {
        // 构建新帧
        current_frame_->features_left_.push_back(
                Feature::Ptr(new Feature(current_frame_,kp)));
        cnt_detected ++;
    }
    LOG(INFO) << "Detect " << cnt_detected << " new features";
    return cnt_detected;
}

/// 找到当前帧右图中与左图所对应的特征 【特征匹配】
int Frontend::FindFeaturesInRight() {
    // use LK flow to estimate points in the right image
    std::vector<cv::Point2f> kps_left,kps_right;
    for(auto &kp:current_frame_->features_left_){
        kps_left.push_back(kp->position_.pt);
        // .lock()方法的功能是：判断weak_ptr所指向的shared_ptr对象是否存在。
        // 若存在，则这个lock方法会返回一个指向该对象的shared_ptr指针；若它所指向的这个shared_ptr对象不存在，则lock()函数会返回一个空的shared_ptr。
        auto mp = kp->map_point_.lock();
        if(mp){
            // use projected points as initial guess
            auto px =
                    camera_right_->world2pixel(mp->pos_,current_frame_->GetPose());
            kps_right.push_back(cv::Point2f(px[0],px[1]));
        }else{
            kps_right.push_back(kp->position_.pt);
        }
    }

    std::vector<uchar> status;
    cv::Mat error;
    // https://blog.csdn.net/weixin_42905141/article/details/93745116
    cv::calcOpticalFlowPyrLK(
                current_frame_->left_img_, current_frame_->right_img_, kps_left,
                kps_right, status, error, cv::Size(11, 11), 3,
                cv::TermCriteria(cv::TermCriteria::COUNT + cv::TermCriteria::EPS, 30,
                                 0.01),
                cv::OPTFLOW_USE_INITIAL_FLOW); // OPTFLOW_USE_INITIAL_FLOW使用初始估计，存储在nextPts中;如果未设置标志，则将prevPts复制到nextPts并将其视为初始估计。

    int num_good_pts =0;
    for(size_t i=0;i< status.size();i++){
        if(status[i]){
            cv::KeyPoint kp(kps_right[i],7); //size : 该关键点邻域直径大小；
            Feature::Ptr feat(new Feature(current_frame_, kp));
            feat->is_on_left_image_ = false;
            current_frame_->features_right_.push_back(feat);
            num_good_pts++;
        }  else {
            current_frame_->features_right_.push_back(nullptr);
        }
    }
        LOG(INFO) << "Find " << num_good_pts << " in the right image.";
        return num_good_pts;
}

/// build init map with single image
/// 使用三角化  生成 MapPoint
/// 后端更新map (Backend)
bool Frontend::BuildInitMap() {
    std::vector<Sophus::SE3d> poses{
        camera_left_->pose(), camera_right_->pose()
    };

    size_t cnt_init_landmarks = 0;

    for(size_t i=0;i<current_frame_->features_left_.size();i++){
        if (current_frame_->features_right_[i] == nullptr) continue;
        // create map point from triangulation
        std::vector<Vec3> points{
                camera_left_->pixel2camera(
                        Vec2(current_frame_->features_left_[i]->position_.pt.x,
                             current_frame_->features_left_[i]->position_.pt.y)),
                camera_right_->pixel2camera(
                        Vec2(current_frame_->features_right_[i]->position_.pt.x,
                             current_frame_->features_right_[i]->position_.pt.y))};

        Vec3 pworld = Vec3::Zero();
        if(triangulation(poses,points,pworld) && pworld[2]>0){
            auto new_map_point = MapPoint::CreateNewMappoint(); // 构造一个new mappoint
            new_map_point->SetPos(pworld);
            new_map_point->AddObservation(current_frame_->features_left_[i]);
            new_map_point->AddObservation(current_frame_->features_right_[i]);

            current_frame_->features_left_[i]->map_point_ = new_map_point;
            current_frame_->features_right_[i]->map_point_ = new_map_point;

            cnt_init_landmarks++;
            map_->InsertMapPoint(new_map_point);
        }
    }
        current_frame_->SetKeyFrame();
        map_->InsertKeyFrame(current_frame_);
        backend_->UpdateMap();
        LOG(INFO) << "Initial map created with " << cnt_init_landmarks
                  << " map points";

        return true;
}

bool Frontend::Reset() {
        LOG(INFO) << "Reset is not implemented. ";
        return true;
    }

}