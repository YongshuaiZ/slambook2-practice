//
// Created by coolas on 23-1-19.
//
#include <opencv2/opencv.hpp>
#include <string>
#include <chrono>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;

string file_1 = "/home/coolas/slambook2-practice/ch8/LK1.png";  // first image
string file_2 = "/home/coolas/slambook2-practice/ch8/LK2.png";  // second image

/// Optical flow tracker and interface
class OpticalFlowTracker{
public:
    OpticalFlowTracker(
            const cv::Mat &img1_,
            const cv::Mat &img2_,
            const vector<cv::KeyPoint> &kp1_,
            vector<cv::KeyPoint> &kp2_,
            vector<bool> &success_,
            bool inverse_ = true, bool has_initial_ = false) :
            img1(img1_), img2(img2_), kp1(kp1_), kp2(kp2_), success(success_), inverse(inverse_),
            has_initial(has_initial_) {}

    void calculateOpticalFlow(const cv::Range &range);

private:
    const cv::Mat &img1;
    const cv::Mat &img2;
    const vector<cv::KeyPoint> &kp1;
    vector<cv::KeyPoint> &kp2;
    vector<bool> &success;
    bool inverse = true;
    bool has_initial = false;
};


/**
 * single level optical flow
 * @param [in] img1 the first image
 * @param [in] img2 the second image
 * @param [in] kp1 keypoints in img1
 * @param [in|out] kp2 keypoints in img2, if empty, use initial guess in kp1
 * @param [out] success true if a keypoint is tracked successfully
 * @param [in] inverse use inverse formulation?
 */
 void OpticalFlowSingleLevel(
         const cv::Mat &img1,
         const cv::Mat &img2,
         const vector<cv::KeyPoint> &kp1,
         vector<cv::KeyPoint> &kp2,
         vector<bool> &success,
         bool inverse = false,
         bool has_initial_guess = false
         );

/**
 * get a gray scale value from reference image (bi-linear interpolated)
 * @param img
 * @param x
 * @param y
 * @return the interpolated value of this pixel
 */

inline float GetPixelValue(const cv::Mat &img, float x, float y) {
    // boundary check
    if (x < 0) x = 0;
    if (y < 0) y = 0;
    if (x >= img.cols - 1) x = img.cols - 2;
    if (y >= img.rows - 1) y = img.rows - 2;

    float xx = x - floor(x); // delt x
    float yy = y - floor(y); // delt y
    int x_a1 = std::min(img.cols - 1, int(x) + 1);
    int y_a1 = std::min(img.rows - 1, int(y) + 1);

    return (1 - xx) * (1 - yy) * img.at<uchar>(y, x)
           + xx * (1 - yy) * img.at<uchar>(y, x_a1)
           + (1 - xx) * yy * img.at<uchar>(y_a1, x)
           + xx * yy * img.at<uchar>(y_a1, x_a1);
}


int main()
{
    // images, note they are CV_8UC1, not CV_8UC3
    cv::Mat img1 = cv::imread(file_1, 0);
    cv::Mat img2 = cv::imread(file_2, 0);

    // key points, using GFTT here.
    vector<cv::KeyPoint> kp1;
    cv::Ptr<cv::GFTTDetector> detector = cv::GFTTDetector::create(500, 0.01, 20); // maximum 500 keypoints
    detector->detect(img1, kp1);

    /// use opencv's flow for validation
    /**
    vector<cv::Point2f> pt1, pt2;
    for (auto &kp: kp1) pt1.push_back(kp.pt);
    vector<uchar> status;
    vector<float> error;
    cv::calcOpticalFlowPyrLK(img1, img2, pt1, pt2, status, error);

    //  绘图
    cv::Mat img2_CV;
    cv::cvtColor(img2, img2_CV, cv::COLOR_GRAY2BGR);
    for (int i = 0; i < pt2.size(); i++) {
        if (status[i]) {
            cv::circle(img2_CV, pt2[i], 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_CV, pt1[i], pt2[i], cv::Scalar(0, 250, 0));
        }
    }

    cv::imshow("tracked by opencv", img2_CV);
    cv::waitKey(0);
    */

    /// single flow
    vector<cv::KeyPoint> kp2_single;
    vector<bool> success_single;
    OpticalFlowSingleLevel(img1, img2, kp1, kp2_single, success_single);
    cv::Mat img2_single;
    cv::cvtColor(img2,img2_single,cv::COLOR_GRAY2BGR);
    for (int i = 0; i < kp2_single.size(); i++) {
        if (success_single[i]) {
            cv::circle(img2_single, kp2_single[i].pt, 2, cv::Scalar(0, 250, 0), 2);
            cv::line(img2_single, kp1[i].pt, kp2_single[i].pt, cv::Scalar(0, 250, 0));
        }
    }

    cv::imshow("tracked single level", img2_single);
    cv::waitKey(0);
}
void OpticalFlowSingleLevel(
        const cv::Mat &img1,
        const cv::Mat &img2,
        const vector<cv::KeyPoint> &kp1,
        vector<cv::KeyPoint> &kp2,
        vector<bool> &success,
        bool inverse,
        bool has_initial
){
    kp2.resize(kp1.size());
    success.resize(kp1.size());
    OpticalFlowTracker tracker(img1,img2,kp1,kp2,success,inverse,has_initial);
    parallel_for_(cv::Range(0, kp1.size()),
                  std::bind(&OpticalFlowTracker::calculateOpticalFlow, &tracker, placeholders::_1));
}

void OpticalFlowTracker::calculateOpticalFlow(const cv::Range &range) {

    // parameters
    int half_patch_size = 4;
    int iterations = 10;
    for(size_t i = range.start; i < range.end; i++)
    {
        auto kp = kp1[i];
        double dx = 0, dy = 0; // dx,dx need to be estimated
        if(has_initial){
            dx = kp2[i].pt.x - kp.pt.x;
            dy = kp2[i].pt.y - kp.pt.y;
        }

        double cost = 0,lastCost = 0 ;
        bool succ = true; //indicate if this point succeeded
        // Gauss-Newton iterations
        Eigen::Matrix2d H = Eigen::Matrix2d::Zero();    // hessian
        Eigen::Vector2d b = Eigen::Vector2d::Zero();    // bias
        Eigen::Vector2d J;  // jacobian
        for(int iter=0;iter< iterations; iter++)
        {
            if(inverse == false)
            {
                // 普通光流法H和b都要更新
                H = Eigen::Matrix2d::Zero();
                b = Eigen::Vector2d::Zero();
            }else
                b = Eigen::Vector2d::Zero();

            cost = 0;
            // compute cost and jacobian
            for (int x = -half_patch_size; x < half_patch_size; x++)
                for (int y = -half_patch_size; y < half_patch_size; y++) {
                    double error = GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y) -
                                   GetPixelValue(img2, kp.pt.x + x + dx, kp.pt.y + y + dy);;  // Jacobian
                    if (inverse == false) {
                        J = -1.0 * Eigen::Vector2d(
                                0.5 * (GetPixelValue(img2, kp.pt.x + dx + x + 1, kp.pt.y + dy + y) -
                                       GetPixelValue(img2, kp.pt.x + dx + x - 1, kp.pt.y + dy + y)),
                                0.5 * (GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y + 1) -
                                       GetPixelValue(img2, kp.pt.x + dx + x, kp.pt.y + dy + y - 1))
                        );
                    } else if (iter == 0) {
                        // in inverse mode, J keeps same for all iterations
                        // NOTE this J does not change when dx, dy is updated, so we can store it and only compute error
                        J = -1.0 * Eigen::Vector2d(
                                0.5 * (GetPixelValue(img1, kp.pt.x + x + 1, kp.pt.y + y) -
                                       GetPixelValue(img1, kp.pt.x + x - 1, kp.pt.y + y)),
                                0.5 * (GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y + 1) -
                                       GetPixelValue(img1, kp.pt.x + x, kp.pt.y + y - 1))
                        );
                    }
                    // compute H, b and set cost;
                    b += -error * J;
                    cost += error * error;
                    if (inverse == false || iter == 0) {
                        // also update H
                        H += J * J.transpose();
                    }
                }
            // compute update
            Eigen::Vector2d update = H.ldlt().solve(b);

            if (std::isnan(update[0])) {
                // sometimes occurred when we have a black or white patch and H is irreversible
                cout << "update is nan" << endl;
                succ = false;
                break;
            }

            if (iter > 0 && cost > lastCost) {
                break;
            }

            // update dx, dy
            dx += update[0];
            dy += update[1];
            lastCost = cost;
            succ = true;

            if (update.norm() < 1e-2) {
                // converge
                break;
            }
        }
        success[i] = succ;

        // set kp2
        kp2[i].pt = kp.pt + cv::Point2f(dx, dy);

    }

}