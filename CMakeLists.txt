cmake_minimum_required(VERSION 3.8)
project(path_planning)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(OpenCV REQUIRED)

# Bridge + RRT library (no ROS)
# Use OpenCV_LIBS / OpenCV_INCLUDE_DIRS for compatibility with Ubuntu's libopencv-dev (OpenCV config may not export OpenCV:: targets)
add_library(path_planning_core
  src/occ_grid_bridge.cpp
  src/rrt_planner.cpp
)
target_include_directories(path_planning_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${OpenCV_INCLUDE_DIRS}
)
target_link_libraries(path_planning_core ${OpenCV_LIBS})

# Path planner ROS2 node
add_executable(path_planner_node src/path_planner_node.cpp)
target_include_directories(path_planner_node PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(path_planner_node path_planning_core)
ament_target_dependencies(path_planner_node rclcpp sensor_msgs geometry_msgs nav_msgs)

install(TARGETS path_planner_node
  DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS
  scripts/visualize_waypoints_occ.py
  DESTINATION lib/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
