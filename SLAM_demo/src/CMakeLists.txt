add_library(myslam SHARED
        frame.cpp
        feature.cpp
        mappoint.cpp
        map.cpp
        camera.cpp
        viewer.cpp
        backend.cpp
        visual_odometry.cpp
        config.cpp
        dataset.cpp
        frontend.cpp)

target_link_libraries(myslam ${THIRD_PARTY_LIBS})