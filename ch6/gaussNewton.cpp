#include <iostream>
#include <chrono>
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main()
{
    double ar = 1.0, br = 2.0, cr = 1.0;         // 真实参数值
    double ae = 2.0, be = -1.0, ce = 5.0;        // 估计参数值
    int N = 100;                                 // 数据点
    double w_sigma = 1.0;                        // 噪声Sigma值
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng;                                 // OpenCV随机数产生器

    /// 构造模拟数据
    vector<double> x_data;
    vector<double> y_data;
    for(int i=0;i<N;i++)
    {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x+ br * x + cr)+rng.gaussian(w_sigma*w_sigma));
    }

    // 开始Gauss-Newton迭代
    int iterations = 100;    // 迭代次数
    double cost = 0, lastCost = 0;  // 本次迭代的cost和上一次迭代的cost

    for(int iter=0;iter<100;iter++)
    {
        // 定义黑塞矩阵，bias
        Matrix3d H = Matrix3d ::Zero(); // Hessian = J^T W^{-1} J in Gauss-Newton
        Vector3d b = Vector3d ::Zero();
        cost = 0;

        for(int i=0;i<N;i++)
        {
            double xi = x_data[i],yi = y_data[i];
            double error = y_data[i] - exp(ae * x_data[i] * x_data[i] + be * x_data[i] + ce) ;
            Vector3d J = Vector3d ::Zero(); // 雅各比矩阵

            J[0] = -xi * xi * exp(ae * xi * xi + be * xi + ce);  // de/da
            J[1] = -xi * exp(ae * xi * xi + be * xi + ce);  // de/db
            J[2] = -exp(ae * xi * xi + be * xi + ce);  // de/dc

            H += inv_sigma * inv_sigma * J * J.transpose();
            b += - inv_sigma * inv_sigma * error * J;
            cost += error * error;
        }

        // 求解线性方程
        Vector3d dx = H.ldlt().solve(b);
        //dx = H.colPivHouseholderQr().solve(b);    //QR分解，可加快求解速度
        //dx = H.ldlt().solve(b);    //ldlt分解，可加快求解速度

        if (isnan(dx[0])) {
            cout << "result is nan!" << endl;
            break;
        }
        if (iter > 0 && cost >= lastCost) {
            cout << "cost: " << cost << ">= last cost: " << lastCost << ", break." << endl;
            break;
        }
        ae = ae + dx[0];
        be = be + dx[1];
        ce = ce + dx[2];

        lastCost = cost;
    }
    cout << "estimated abc = " << ae << ", " << be << ", " << ce << endl;
    return 0;
}