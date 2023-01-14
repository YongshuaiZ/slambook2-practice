//
// Created by coolas on 23-1-14.
//
#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/features2d/features2d.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <chrono>

using namespace std;

string image1_path = "/home/coolas/slambook2-practice/ch7/1.png";
string image2_path = "/home/coolas/slambook2-practice/ch7/2.png";

int main()
{
    //读取文件
    cv::Mat image1,image2;
    image1 = cv::imread(image1_path,CV_LOAD_IMAGE_COLOR);
    image2 = cv::imread(image2_path,CV_LOAD_IMAGE_COLOR);
    assert(image1.data != nullptr && image2.data != nullptr);

    //初始化 角点，描述子，特征匹配
    vector<cv::KeyPoint> keypoints_1,keypoints_2;
    cv::Mat descrip_1,descrip_2;

    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create(); // 角点检测
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create(); // 特征点检测
    cv::Ptr<cv::DescriptorMatcher> matcher = cv::DescriptorMatcher::create("BruteForce-Hamming");

    //1. 计算关键点
    detector->detect(image1,keypoints_1);
    detector->detect(image2,keypoints_2);

    //2. 根据关键点计算 描述子
    descriptor->compute(image1,keypoints_1,descrip_1);  //descrip_1 : keypoints.size() * 32 (关键点个数 * 32维）
    descriptor->compute(image2,keypoints_2,descrip_2);

    cout<<"descrip_1 "<<descrip_1.rows<<' '<<descrip_1.<<endl;
    cout<<"keypoints" <<keypoints_1.size()<<endl;

    for(int i=0;i<descrip_1.rows;i++)
    {
        for(int j=0;j<descrip_2.cols;j++)
        {
            cout<<descrip_1.at<bool>(i,j)<<' ';
        }
        cout<<endl;
    }



    cv::Mat img1_out;
    cv::drawKeypoints(image1,keypoints_1,img1_out,cv::Scalar::all(-1),cv::DrawMatchesFlags::DEFAULT);
    cv::imshow("ORB feature",img1_out);

//    cv::waitKey(0);
    //3. 根据描述子 进行匹配，使用Hamming距离
    vector<cv::DMatch> matches;
    matcher->match(descrip_1,descrip_2,matches);

    //4. 匹配点筛选， 计算最小距离和最大距离
    auto min_max = minmax_element(matches.begin(),matches.end(),
                                  [](const cv::DMatch &m1,const cv::DMatch &m2){return m1.distance<m2.distance; });
    double min_dist = min_max.first->distance;
    double max_dist = min_max.second->distance;

    vector<cv::DMatch> good_matches;
    for(int i=0;i<descrip_1.rows;i++)
    {
        if(matches[i].distance <= max(2*min_dist,30.0));
            good_matches.push_back(matches[i]);
    }
    //5. 绘制匹配结果
    cv::Mat img_match,img_goodmatch;

    cv::drawMatches(image1,keypoints_1,image2,keypoints_2,matches,img_match);
    cv::drawMatches(image1,keypoints_1,image2,keypoints_2,good_matches,img_goodmatch);

    cv::imshow("all matches",img_match);
    cv::imshow("good match",img_goodmatch);
//    cv::waitKey(0);
    return 0;
}