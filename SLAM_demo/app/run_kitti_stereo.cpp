//
// Created by coolas on 23-1-30.
//
#include <gflags/gflags.h>
#include "myslam/visual_odometry.h"

// DEFINE宏包含有三个参数：1、flag的命名；2、flag的默认值；3、该flag对应的一些提示性说明
DEFINE_string(config_file, "../config/default.yaml", "config file path");

int main(int argc, char **argv) {
    // remove_flags，如果为true的话，ParseCommandLineFlags会移除相应的flag和对应的参数并且修改相应的argc，然后argv只会保留命令行参数；
    // 如果remove_flags为false的话，会保持argc不变，但是会调整argv中存储内容的顺序，并且把flag放在命令行参数的前面
    google::ParseCommandLineFlags(&argc, &argv, true);

    myslam::VisualOdometry::Ptr vo(
            new myslam::VisualOdometry(FLAGS_config_file));
    assert(vo->Init() == true);
    vo->Run();

    return 0;
}