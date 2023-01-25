//
// Created by coolas on 23-1-25.
//
#include <iostream>
#include <opencv2/core/core.hpp>
#include <ceres/ceres.h>
#include <chrono>

using namespace std;

// 代价函数的计算模型
struct CURVE_FITTING_COST{
    CURVE_FITTING_COST(double x,double y):_x(x), _y(y){}

    // 残差计算
    template<typename T> // 定义参数块和代价函数
    bool operator()(const T* const abc,T *residual) const {
        residual[0] = T(_y) - ceres::exp(abc[0] * T(_x) * T(_x) + abc[1]*T(_x) + abc[2]); // y - exp(ax^2 + bx + c)
        return true;
    }

    const double _x,_y; // x，y data
};

int main()
{
    double ar = 1.0, br = 2.0, cr = 1.0;         // 真实参数值
    double ae = 2.0, be = -1.0, ce = 5.0;        // 估计参数值
    int N = 100;                                 // 数据点
    double w_sigma = 1.0;                        // 噪声Sigma值
    double inv_sigma = 1.0 / w_sigma;
    cv::RNG rng;                                 // OpenCV随机数产生器

    vector<double> x_data, y_data;      // 数据
    for (int i = 0; i < N; i++) {
        double x = i / 100.0;
        x_data.push_back(x);
        y_data.push_back(exp(ar * x * x + br * x + cr) + rng.gaussian(w_sigma * w_sigma));
    }

    double abc[3] = {ae, be, ce};

    // 构建最小二乘问题
    ceres::Problem problem;
    for(int i=0;i<N;i++){
        problem.AddResidualBlock( //添加误差项
            // 使用自动求导，<误差项 和 优化变量的维度>
                new ceres::AutoDiffCostFunction<CURVE_FITTING_COST,1,3>(
                        new CURVE_FITTING_COST(x_data[i],y_data[i])
                        ),
                nullptr, // 核函数，这里不使用，为空
                abc // 待估计参数
                );
}
    // 配置求解器
    ceres::Solver::Options options;
    options.linear_solver_type = ceres::DENSE_NORMAL_CHOLESKY;
    options.minimizer_progress_to_stdout = true; // 输出到cout

    ceres::Solver::Summary summary; //优化信息

    ceres::Solve(options,&problem,&summary);

    // 输出结果
    cout<<summary.BriefReport()<<endl;
    cout<<"estimated a,b,c";
    for(auto a:abc)
        cout<<a<<" ";
    cout<<endl;

    return 0;
}