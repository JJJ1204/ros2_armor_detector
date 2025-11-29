cmake_minimum_required(VERSION 3.8)
project(aim_detector)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# 1. 寻找依赖库
find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(cv_bridge REQUIRED)
find_package(aim_interfaces REQUIRED)
find_package(OpenCV REQUIRED) # 必须有这一行

# 2. 创建可执行文件
add_executable(detector_node src/detector_node.cpp)

# 3. 链接 ROS 2 专用库 (必须用这个宏，不能用 target_link_libraries)
ament_target_dependencies(detector_node
  rclcpp
  sensor_msgs
  cv_bridge
  aim_interfaces
)

# 4. 链接 OpenCV 库 (必须用这个宏，专门处理第三方库)
target_link_libraries(detector_node
  ${OpenCV_LIBS}
)

# 5. 安装规则
install(TARGETS detector_node
  DESTINATION lib/${PROJECT_NAME})

ament_package()