//
// Created by coolas on 23-1-19.
//
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <Eigen/Core>
//#include <g2o/core/base_vertex.h>
//#include <g2o/core/base_unary_edge.h>
//#include <g2o/core/sparse_optimizer.h>
//#include <g2o/core/block_solver.h>
//#include <g2o/core/solver.h>
//#include <g2o/core/optimization_algorithm_gauss_newton.h>
//#include <g2o/solvers/dense/linear_solver_dense.h>
#include <sophus/se3.hpp>
#include <chrono>

using namespace std;
using namespace cv;

typedef vector<Eigen::Vector2d, Eigen::aligned_allocator<Eigen::Vector2d>> VecVector2d;
typedef vector<Eigen::Vector3d, Eigen::aligned_allocator<Eigen::Vector3d>> VecVector3d;

// BA by gauss-newton
void bundleAdjustmentGaussNewton(
        const VecVector3d &points_3d,
        const VecVector2d &points_2d,
        const Mat &K,
        Sophus::SE3d &pose
);
// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);

int main()
{
    string image1_path = "/home/coolas/SLAMProjects/slambook2/ch7/1.png";
    string image2_path = "/home/coolas/SLAMProjects/slambook2/ch7/2.png";
    string image1_depth_path = "/home/coolas/SLAMProjects/slambook2/ch7/1_depth.png";

    //-- 读取图像
    Mat image1 = imread(image1_path, CV_LOAD_IMAGE_COLOR);
    Mat image2 = imread(image2_path, CV_LOAD_IMAGE_COLOR);
    assert(img_1.data && img_2.data && "Can not load images!");

    // 提取特征点，特征匹配
    vector<KeyPoint> keypoints_1,keypoints_2;
    cv::Mat descrip_1,descrip_2;
    vector<cv::DMatch> matches;

    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(); // 角点检测
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create(); // 特征点检测
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    //1. 计算关键点
    detector->detect(image1,keypoints_1);
    detector->detect(image2,keypoints_2);

    //2. 根据关键点计算 描述子
    descriptor->compute(image1,keypoints_1,descrip_1);  //descrip_1 : keypoints.size() * 32 (关键点个数 * 32维）
    descriptor->compute(image2,keypoints_2,descrip_2);

    // 匹配
    matcher->match(descrip_1,descrip_2,matches);

    // 计算3D点
    Mat image1_depth = cv::imread(image1_depth_path,CV_LOAD_IMAGE_UNCHANGED);
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    vector<cv::Point3f> pts_3d; //
    vector<cv::Point2f> pts_2d;

    for(DMatch m:matches)
    {
        short d = image1_depth.at<unsigned short>(int(keypoints_1[m.queryIdx].pt.y),int(keypoints_1[m.queryIdx].pt.x));
        if (d == 0)   // bad depth
            continue;
        float dd = d / 5000.0;

        Point2d p1 = pixel2cam(keypoints_1[m.queryIdx].pt,K);
        pts_3d.push_back(Point3f(
                p1.x * dd,
                p1.y * dd,
                dd
                ));
        pts_2d.push_back(Point2f (
                keypoints_2[m.trainIdx].pt.x,
                keypoints_2[m.trainIdx].pt.y
        ));
    }
    VecVector3d points_3d;
    VecVector2d points_2d;
    for(int i=0;i<pts_3d.size();i++)
    {
        points_3d.push_back(Eigen::Vector3d(pts_3d[i].x,pts_3d[i].y,pts_3d[i].z));
        points_2d.push_back(Eigen::Vector2d(pts_2d[i].x,pts_2d[i].y));
    }
    Mat r, t;
    solvePnP(pts_3d, pts_2d, K, Mat(), r, t, false); // 调用OpenCV 的 PnP 求解，可选择EPNP，DLS等方法
    Mat R;
    cv::Rodrigues(r, R); // r为旋转向量形式，用Rodrigues公式转换为矩阵

    cout << "R=" << endl << R << endl;
    cout << "t=" << endl << t << endl;
//    Sophus::SE3d pose_gn;
//    bundleAdjustmentGaussNewton(points_3d, points_2d, K, pose_gn);

}
void bundleAdjustmentGaussNewton(
        const VecVector3d &points_3d,
        const VecVector2d &points_2d,
        const Mat &K,
        Sophus::SE3d &pose
){
    typedef Eigen::Matrix<double,6,1> Vector6d;
    const int iterations = 10;
    double cost = 0,lastCost = 0;
    double fx = K.at<double>(0,0);
    double fy = K.at<double>(1,1);
    double cx = K.at<double>(0,2);
    double cy = K.at<double>(1,2);

    for(int iter = 0;iter< iterations;iter++)
    {
        // 定义黑塞矩阵和bias
        Eigen::Matrix<double,6,6> H = Eigen::Matrix<double,6,6>::Zero();
        Vector6d b = Vector6d ::Zero();

        for(int i=0;i<points_3d.size();i++)
        {
            // error = true - estimate
            Eigen::Vector3d pc = pose * points_3d[i]; //camera 2 坐标系下的点
            Eigen::Vector2d proj(fx * pc[0] / pc[2] + cx, fy * pc[1] / pc[2] + cy);

            Eigen::Vector2d e = points_2d[i] - proj;

            double inv_z = 1.0 / pc[2];
            double inv_z2 = inv_z * inv_z;

            Eigen::Matrix<double,2,6> J ;
            J << -fx * inv_z,
                    0,
                    fx * pc[0] * inv_z2,
                    fx * pc[0] * pc[1] * inv_z2,
                    -fx - fx * pc[0] * pc[0] * inv_z2,
                    fx * pc[1] * inv_z,
                    0,
                    -fy * inv_z,
                    fy * pc[1] * inv_z2,
                    fy + fy * pc[1] * pc[1] * inv_z2,
                    -fy * pc[0] * pc[1] * inv_z2,
                    -fy * pc[0] * inv_z;

            H += J.transpose() * J;
            b += -J.transpose() * e;

            cost += e.squaredNorm();
        }

        Vector6d dx;
        dx = H.ldlt().solve(b);

        if (isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }

        if (iter > 0 && cost >= lastCost) {
            // cost increase, update is not good
            cout << "cost: " << cost << ", last cost: " << lastCost << endl;
            break;
        }
        // update your estimation
        pose = Sophus::SE3d::exp(dx) * pose;
        lastCost = cost;

        cout << "iteration " << iter << " cost=" << cost << endl;
        if (dx.norm() < 1e-6) {
            // converge
            break;
        }
    }

    cout << "pose by g-n: \n" << pose.matrix() << endl;

}

Point2d pixel2cam(const Point2d &p, const Mat &K) {
    return Point2d
            (
                    (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                    (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
            );
}
