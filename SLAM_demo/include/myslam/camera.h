//
// Created by coolas on 23-1-26.
//

#ifndef MYSLAM_CAMERA_H
#define MYSLAM_CAMERA_H

#include "myslam/common_include.h"

namespace myslam{

// Pinhole stereo camera model

class Camera{
public:
    EIGEN_MAKE_ALIGNED_OPERATOR_NEW;
    typedef std::shared_ptr<Camera> Ptr;

    double fx_=0 , fy_=0, cx_=0, cy_=0,baseline_ = 0; //Camera intrinsics
    Sophus::SE3d pose_;                               // extrinsic, from stereo camera to single camera
    Sophus::SE3d pose_inv_;                           //inverse of extrinsics

    Camera();
    Camera(double fx, double fy, double cx, double cy, double baseline,
           const Sophus::SE3d &pose)
            : fx_(fx), fy_(fy), cx_(cx), cy_(cy), baseline_(baseline), pose_(pose) {
        pose_inv_ = pose_.inverse();
    }

    Sophus::SE3d pose() const { return pose_; }

    // return intrinsic matrix
    Mat33 K() const {
        Mat33 k;
        k << fx_, 0, cx_, 0, fy_, cy_, 0, 0, 1;
        return k;
    }

    // coordinate transform: world, camera, pixel
    Vec3 world2camera(const Vec3 &p_w, const Sophus::SE3d &T_c_w);

    Vec3 camera2world(const Vec3 &p_c, const Sophus::SE3d &T_c_w);

    Vec2 camera2pixel(const Vec3 &p_c);

    Vec3 pixel2camera(const Vec2 &p_p, double depth = 1);

    Vec3 pixel2world(const Vec2 &p_p, const Sophus::SE3d &T_c_w, double depth = 1);

    Vec2 world2pixel(const Vec3 &p_w, const Sophus::SE3d &T_c_w);
};

}
#endif //MYSLAM_CAMERA_H
