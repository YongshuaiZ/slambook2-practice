#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/calib3d/calib3d.hpp>
#include <opencv2/imgproc.hpp>

// #include "extra.h" // use this if in OpenCV2

using namespace std;
using namespace cv;

/****************************************************
 * 本程序演示了如何使用2D-2D的特征匹配估计相机运动
 * **************************************************/

void find_feature_matches(
        const Mat &img_1, const Mat &img_2,
        std::vector<KeyPoint> &keypoints_1,
        std::vector<KeyPoint> &keypoints_2,
        std::vector<DMatch> &matches);

void pose_estimation_2d2d(
        std::vector<KeyPoint> keypoints_1,
        std::vector<KeyPoint> keypoints_2,
        std::vector<DMatch> matches,
        Mat &R, Mat &t);

// 像素坐标转相机归一化坐标
Point2d pixel2cam(const Point2d &p, const Mat &K);

void triangulation(
        const vector<KeyPoint> &keypoint_1,
        const vector<KeyPoint> &keypoint_2,
        const std::vector<DMatch> &matches,
        const Mat &R, const Mat &t,
        vector<Point3d> &points);

/// 作图用
inline cv::Scalar get_color(float depth) {
    float up_th = 50, low_th = 10, th_range = up_th - low_th;
    if (depth > up_th) depth = up_th;
    if (depth < low_th) depth = low_th;
    return cv::Scalar(255 * depth / th_range, 0, 255 * (1 - depth / th_range));
}


int main() {
    string image1_path = "/home/coolas/SLAMProjects/slambook2/ch7/1.png";
    string image2_path = "/home/coolas/SLAMProjects/slambook2/ch7/2.png";
    //-- 读取图像
    Mat img_1 = imread(image1_path, CV_LOAD_IMAGE_COLOR);
    Mat img_2 = imread(image2_path, CV_LOAD_IMAGE_COLOR);
    assert(img_1.data && img_2.data && "Can not load images!");

    vector<KeyPoint> keypoints_1, keypoints_2;
    vector<DMatch> matches;
    find_feature_matches(img_1, img_2, keypoints_1, keypoints_2, matches);
    cout << "一共找到了" << matches.size() << "组匹配点" << endl;

    //-- 估计两张图像间运动
    Mat R, t;
    pose_estimation_2d2d(keypoints_1, keypoints_2, matches, R, t);

    //-- 验证E=t^R*scale
    // 手写t的反对称矩阵，这里利用了OpenCV的Mat的子类 Mat_
    // Mat_可以像Eigen中的matrix一样，用<<流操作 和 ,逗号分隔符进行初始化
    // Mat_重载了许多操作符，以供方便
    Mat t_x =
            (Mat_<double>(3, 3) << 0, -t.at<double>(2, 0), t.at<double>(1, 0),
                    t.at<double>(2, 0), 0, -t.at<double>(0, 0),
                    -t.at<double>(1, 0), t.at<double>(0, 0), 0);

    cout << "t^R=" << endl << t_x * R << endl;

    //-- 验证对极约束
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);
    for (DMatch m: matches) {
        Point2d pt1 = pixel2cam(keypoints_1[m.queryIdx].pt, K);
        Mat y1 = (Mat_<double>(3, 1) << pt1.x, pt1.y, 1);
        Point2d pt2 = pixel2cam(keypoints_2[m.trainIdx].pt, K);
        Mat y2 = (Mat_<double>(3, 1) << pt2.x, pt2.y, 1);
        Mat d = y2.t() * t_x * R * y1;
//        cout << "epipolar constraint = " << d << endl;
    }
    vector<Point3d> points; //空间三维坐标
    triangulation(keypoints_1,keypoints_2,matches,R,t,points);

    //-- 验证三角化点与特征点的重投影关系
    Mat img1_plot = img_1.clone();
    Mat img2_plot = img_2.clone();
    for (int i = 0; i < matches.size(); i++) {
        // 第一个图
        float depth1 = points[i].z;
        cout << "depth: " << depth1 << endl;
        Point2d pt1_cam = pixel2cam(keypoints_1[matches[i].queryIdx].pt, K);
        cv::circle(img1_plot, keypoints_1[matches[i].queryIdx].pt, 2, get_color(depth1), 2);

        // 第二个图
        Mat pt2_trans = R * (Mat_<double>(3, 1) << points[i].x, points[i].y, points[i].z) + t;
        float depth2 = pt2_trans.at<double>(2, 0);
        cv::circle(img2_plot, keypoints_2[matches[i].trainIdx].pt, 2, get_color(depth2), 2);
    }
    cv::imshow("img 1", img1_plot);
    cv::imshow("img 2", img2_plot);
    cv::waitKey();

    return 0;
}

void find_feature_matches(const Mat &img_1, const Mat &img_2,
                          std::vector<KeyPoint> &keypoints_1,
                          std::vector<KeyPoint> &keypoints_2,
                          std::vector<DMatch> &matches) {
    //-- 初始化
    Mat descriptors_1, descriptors_2;
    // used in OpenCV3
    Ptr<FeatureDetector> detector = ORB::create();
    Ptr<DescriptorExtractor> descriptor = ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    Ptr<DescriptorMatcher> matcher = DescriptorMatcher::create("BruteForce-Hamming");
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect(img_1, keypoints_1);
    detector->detect(img_2, keypoints_2);

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute(img_1, keypoints_1, descriptors_1);
    descriptor->compute(img_2, keypoints_2, descriptors_2);

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    vector<DMatch> match;
    //BFMatcher matcher ( NORM_HAMMING );
    matcher->match(descriptors_1, descriptors_2, match);

    //-- 第四步:匹配点对筛选
    double min_dist = 10000, max_dist = 0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for (int i = 0; i < descriptors_1.rows; i++) {
        double dist = match[i].distance;
        if (dist < min_dist) min_dist = dist;
        if (dist > max_dist) max_dist = dist;
    }

    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for (int i = 0; i < descriptors_1.rows; i++) {
        if (match[i].distance <= max(2 * min_dist, 30.0)) {
            matches.push_back(match[i]);
        }
    }
}

Point2d pixel2cam(const Point2d &p, const Mat &K) {
    return Point2d
            (
                    (p.x - K.at<double>(0, 2)) / K.at<double>(0, 0),
                    (p.y - K.at<double>(1, 2)) / K.at<double>(1, 1)
            );
}

void pose_estimation_2d2d(std::vector<KeyPoint> keypoints_1,
                          std::vector<KeyPoint> keypoints_2,
                          std::vector<DMatch> matches,
                          Mat &R, Mat &t) {
    // 相机内参,TUM Freiburg2
    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    //-- 把匹配点转换为vector<Point2f>的形式
    vector<Point2f> points1;
    vector<Point2f> points2;

    for (int i = 0; i < (int) matches.size(); i++) {
        points1.push_back(keypoints_1[matches[i].queryIdx].pt);
        points2.push_back(keypoints_2[matches[i].trainIdx].pt);
    }

    //-- 计算基础矩阵
    Mat fundamental_matrix;
    fundamental_matrix = findFundamentalMat(points1, points2, CV_FM_8POINT);
    cout << "fundamental_matrix is " << endl << fundamental_matrix << endl;

    //-- 计算本质矩阵
    //    //-- 计算本质矩阵
    //    // 注意计算本质矩阵需要说明内参，这里给到光心位置和焦距等价于给出内参
    //    // 因为通过像素坐标的对极约束公式p2^T * F * p1我们直接计算的应该是F矩阵，
    //    // 进而给出内参K，求出E矩阵（OpenCV自动筛选两个角度深度值均为正的结果）
    Point2d principal_point(325.1, 249.7);  //相机光心, TUM dataset标定值
    double focal_length = 521;      //相机焦距, TUM dataset标定值
    Mat essential_matrix;
    essential_matrix = findEssentialMat(points1, points2, focal_length, principal_point);
    cout << "essential_matrix is " << endl << essential_matrix << endl;

    //-- 计算单应矩阵
    //-- 但是本例中场景不是平面，单应矩阵意义不大
    Mat homography_matrix;
    homography_matrix = findHomography(points1, points2, RANSAC, 3);
    cout << "homography_matrix is " << endl << homography_matrix << endl;

    //-- 从本质矩阵中恢复旋转和平移信息.
    // 此函数仅在Opencv3中提供
    recoverPose(essential_matrix, points1, points2, R, t, focal_length, principal_point);
    cout << "R is " << endl << R << endl;
    cout << "t is " << endl << t << endl;

}
void triangulation(
        const vector<KeyPoint> &keypoint_1,
        const vector<KeyPoint> &keypoint_2,
        const std::vector<DMatch> &matches,
        const Mat &R, const Mat &t,
        vector<Point3d> &points)
{
    Mat T1  = (Mat_<float>(3,4)
            << 1 , 0 , 0 , 0,
               0 , 1 , 0 , 0,
               0 , 0 , 1 , 0);

    Mat T2 = (Mat_<float>(3,4)
            << R.at<double>(0,0),R.at<double>(0,1),R.at<double>(0,2),t.at<double>(0,0),
               R.at<double>(1,0),R.at<double>(1,1),R.at<double>(1,2),t.at<double>(1,0),
               R.at<double>(2,0),R.at<double>(2,1),R.at<double>(2,2),t.at<double>(2,0));

    Mat K = (Mat_<double>(3, 3) << 520.9, 0, 325.1, 0, 521.0, 249.7, 0, 0, 1);

    vector<cv::Point2f> pts_1,pts_2;
    // 需要将像素坐标系下坐标转换成相机坐标系
    for(DMatch m:matches)
    {
        pts_1.push_back(pixel2cam(keypoint_1[m.queryIdx].pt,K));
        pts_2.push_back(pixel2cam(keypoint_2[m.trainIdx].pt,K));
    }

    Mat pts_4d;
    cv::triangulatePoints(T1,T2,pts_1,pts_2,pts_4d);

    // 将齐次 转换成非齐次
    cout<<"rows: "<< pts_4d.rows << "  cols: " << pts_4d.cols;

    for(int i=0;i<pts_4d.cols;i++)
    {
        Mat x = pts_4d.col(i);
        x = x / x.at<float>(3,0);
        Point3d p(
                x.at<float>(0, 0),
                x.at<float>(1, 0),
                x.at<float>(2, 0)
        );
        points.push_back(p);
    }

}
