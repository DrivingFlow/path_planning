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
find_package(std_msgs REQUIRED)
find_package(tf2 REQUIRED)
find_package(OpenCV REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(rosidl_default_generators REQUIRED)

# Custom messages: array of occupancy grids; agent-centered binary input
rosidl_generate_interfaces(${PROJECT_NAME}
  "msg/OccupancyGridArray.msg"
  "msg/AgentCenteredInput.msg"
  DEPENDENCIES std_msgs nav_msgs
)

# Bridge + RRT library (no ROS)
# Use OpenCV_LIBS / OpenCV_INCLUDE_DIRS for compatibility with Ubuntu's libopencv-dev (OpenCV config may not export OpenCV:: targets)
add_library(path_planning_core
  src/occ_grid_bridge.cpp
  src/rrt_planner.cpp
  src/astar_energy_planner.cpp
)
target_include_directories(path_planning_core PUBLIC
  ${CMAKE_CURRENT_SOURCE_DIR}/src
  ${OpenCV_INCLUDE_DIRS}
  ${EIGEN3_INCLUDE_DIR}
)
target_link_libraries(path_planning_core ${OpenCV_LIBS})

# Path planner ROS2 node
add_executable(path_planner_node src/path_planner_node.cpp)
target_include_directories(path_planner_node PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src)
target_link_libraries(path_planner_node path_planning_core)
ament_target_dependencies(path_planner_node rclcpp sensor_msgs geometry_msgs nav_msgs std_msgs tf2)
rosidl_get_typesupport_target(cpp_typesupport_target ${PROJECT_NAME} "rosidl_typesupport_cpp")
target_link_libraries(path_planner_node ${cpp_typesupport_target})

add_executable(visualize_waypoints_occ_cpp scripts/visualize_waypoints_occ.cpp)
target_include_directories(visualize_waypoints_occ_cpp PRIVATE ${CMAKE_CURRENT_SOURCE_DIR}/src ${EIGEN3_INCLUDE_DIR})
target_link_libraries(visualize_waypoints_occ_cpp ${OpenCV_LIBS})
ament_target_dependencies(visualize_waypoints_occ_cpp rclcpp nav_msgs geometry_msgs)

install(TARGETS path_planner_node
  DESTINATION lib/${PROJECT_NAME})

install(TARGETS visualize_waypoints_occ_cpp
  DESTINATION lib/${PROJECT_NAME})

install(PROGRAMS
  scripts/visualize_waypoints_occ.py
  DESTINATION lib/${PROJECT_NAME}
)

install(DIRECTORY launch
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY msg
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
endif()

ament_package()
