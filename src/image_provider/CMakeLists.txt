cmake_minimum_required(VERSION 3.8)
project(image_provider)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(image_transport REQUIRED)

# =========================================================
# 🔴 复制之前的核弹修复：强制指定系统 OpenCV
# =========================================================
set(SYSTEM_OPENCV_INCLUDE_DIR "/usr/include/opencv4")
set(SYSTEM_OPENCV_LIBS
  "/usr/lib/x86_64-linux-gnu/libopencv_core.so"
  "/usr/lib/x86_64-linux-gnu/libopencv_imgproc.so"
  "/usr/lib/x86_64-linux-gnu/libopencv_highgui.so"
  "/usr/lib/x86_64-linux-gnu/libopencv_imgcodecs.so" 
)
# =========================================================

add_executable(image_publisher_node src/image_publisher.cpp)

target_include_directories(image_publisher_node PUBLIC
  $<BUILD_INTERFACE:${CMAKE_CURRENT_SOURCE_DIR}/include>
  $<INSTALL_INTERFACE:include>
  ${SYSTEM_OPENCV_INCLUDE_DIR}
)

target_link_libraries(image_publisher_node
  ${SYSTEM_OPENCV_LIBS}
  cv_bridge::cv_bridge
)

ament_target_dependencies(image_publisher_node
  rclcpp
  sensor_msgs
  image_transport
)

install(TARGETS image_publisher_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()