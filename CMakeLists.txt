project(realsense)
set(VERSION 1.0.0)

cmake_minimum_required(VERSION 3.12)
set(CMAKE_CXX_STANDARD 11)

add_executable(rs-rgbd-viewer rs-rgbd-viewer/rs-rgbd-viewer.cpp)
target_compile_options(rs-rgbd-viewer PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -pedantic -Werror -g -ggdb>
)
install(TARGETS rs-rgbd-viewer)
