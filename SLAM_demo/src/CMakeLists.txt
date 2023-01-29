add_library(myslam SHARED
        frame.cpp
        feature.cpp
        mappoint.cpp
        map.cpp
        frontend.cpp
        camera.cpp
        viewer.cpp)

target_link_libraries(myslam ${THIRD_PARTY_LIBS})