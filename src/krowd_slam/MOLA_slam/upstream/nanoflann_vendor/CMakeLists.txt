cmake_minimum_required(VERSION 3.16)
project(nanoflann_vendor VERSION 1.11.0)

find_package(ament_cmake REQUIRED)

# Mirror the layout of the ros-humble-nanoflann Debian package so the
# relocatable nanoflannConfig.cmake resolves <prefix>/include from
# <prefix>/share/cmake/nanoflann/.
install(FILES include/nanoflann.hpp DESTINATION include)
install(
  FILES
    cmake/nanoflannConfig.cmake
    cmake/nanoflannConfigVersion.cmake
    cmake/nanoflannTargets.cmake
  DESTINATION share/cmake/nanoflann
)

ament_package()
