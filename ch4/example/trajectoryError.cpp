//
// Created by coolas on 23-1-11.
//
#include <iostream>
#include <fstream>
#include <unistd.h>
#include <pangolin/pangolin.h>
#include <sophus/se3.hpp>

using namespace Sophus;
using namespace std;

string groundtruth_path = "/home/coolas/slambook2-practice/ch4/example/groundtruth.txt";
string estimated_path = "/home/coolas/slambook2-practice/ch4/example/estimated.txt";

typedef vector<Sophus::SE3d, Eigen::aligned_allocator<Sophus::SE3d>> TrajectoryType;

TrajectoryType ReadTrajectory(const string &path);

int main()
{
    TrajectoryType groundtruth = ReadTrajectory(groundtruth_path);
    TrajectoryType estimated = ReadTrajectory(estimated_path);

    double rmse;
    for(size_t i=0;i<groundtruth.size();i++)
    {
        double error = (groundtruth[i].inverse()*estimated[i]).log().norm();
        rmse += error * error;
    }
    rmse = rmse / double(estimated.size());
    rmse = sqrt(rmse);
    cout << "RMSE = "<< rmse<<endl;

    return 0;
}
TrajectoryType ReadTrajectory(const string &path)
{
    ifstream fin(path);
    TrajectoryType trajectory;
    if(!fin)
    {
        cout<<"读取文件失败";
        return trajectory;
    }
    while(!fin.eof())
    {
        double time, tx, ty, tz, qx, qy, qz, qw;
        fin >> time >> tx >> ty >> tz >> qx >> qy >> qz >> qw;
        Sophus::SE3d p1(Eigen::Quaterniond(qw, qx, qy, qz), Eigen::Vector3d(tx, ty, tz));
        trajectory.push_back(p1);
    }
    return trajectory;
}