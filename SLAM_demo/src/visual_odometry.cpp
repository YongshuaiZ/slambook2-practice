//
// Created by coolas on 23-1-30.
//

#include "myslam/visual_odometry.h"
#include <chrono>
#include "myslam/config.h"
#include "myslam/dataset.h"

namespace myslam{
    VisualOdometry::VisualOdometry(std::string &config_path)
            : config_file_path_(config_path) {}

    bool VisualOdometry::Init() {
        // use [cv::FileStorage] to read from config file
        if (Config::SetParameterFile(config_file_path_) == false) {
            return false;
        }

        dataset_ = Dataset::Ptr(new Dataset(Config::Get<std::string>("dataset_dir")));
        CHECK_EQ(dataset_->Init(), true); // 读取内外参

        // create components and links
        frontend_ = Frontend::Ptr(new Frontend);
        backend_ = Backend::Ptr(new Backend);
        map_ = Map::Ptr(new Map);
        viewer_ = Viewer::Ptr(new Viewer);

        frontend_->SetBackend(backend_);
        frontend_->SetMap(map_);
        frontend_->SetViewer(viewer_);
        frontend_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1)); // 在datasets->Init()对camera_进行赋值

        backend_->SetMap(map_);
        backend_->SetCameras(dataset_->GetCamera(0), dataset_->GetCamera(1));

        viewer_->SetMap(map_);

        return true;
    }

    void VisualOdometry::Run() {
        while (1) {
            LOG(INFO) << "VO is running";
            if (Step() == false) {
                break;
            }
        }

        backend_->Stop();
        viewer_->Close();

        LOG(INFO) << "VO exit";
    }

    bool VisualOdometry::Step() {

        Frame::Ptr new_frame = dataset_->NextFrame(); // 构造帧Frame，该帧中有【ID，左图和右图】

        if (new_frame == nullptr) return false;
        LOG(INFO) << "duan dian";
        auto t1 = std::chrono::steady_clock::now();
        bool success = frontend_->AddFrame(new_frame);  // 进入前端
        auto t2 = std::chrono::steady_clock::now();
        auto time_used =
                std::chrono::duration_cast<std::chrono::duration<double>>(t2 - t1);
        LOG(INFO) << "VO cost time: " << time_used.count() << " seconds.";
        return success;
    }
}