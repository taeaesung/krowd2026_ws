cmake_minimum_required(VERSION 3.16)
project(simplemap_tools)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

find_package(ament_cmake REQUIRED)
find_package(mrpt-maps REQUIRED)
find_package(mrpt-obs REQUIRED)
find_package(mrpt-poses REQUIRED)

add_executable(sm-set-poses src/sm_set_poses.cpp)
target_link_libraries(sm-set-poses mrpt::maps mrpt::obs mrpt::poses)

# bin/, like mp2p_icp_core's sm-cli / sm2mm, so sourcing the workspace puts it
# on PATH for scripts/retraject_map.sh.
install(TARGETS sm-set-poses DESTINATION bin)

ament_package()
