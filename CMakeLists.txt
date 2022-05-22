project(realsense)
set(VERSION 1.0.0)

cmake_minimum_required(VERSION 3.12)
set(CMAKE_MODULE_PATH ${PROJECT_SOURCE_DIR}/cmake)
set(CMAKE_CXX_STANDARD 11)

# set OpenCV_DIR to path containing OpenCVConfig.cmake, e.g., /usr/local/Cellar/opencv/4.5.5_2/lib/cmake/opencv4
find_package(OpenCV REQUIRED)

# Fugly way to find the librealsense library on Windows
if(WIN32)
  if("${CMAKE_SIZEOF_VOID_P}" STREQUAL "8")
    set(LIBRARY_DIR "C:\\Program Files (x86)\\Intel RealSense SDK 2.0\\lib\\x64")
    set(DLL_DIR "C:\\Program Files (x86)\\Intel RealSense SDK 2.0\\bin\\x64")
  else()
    set(LIBRARY_DIR "C:\\Program Files (x86)\\Intel RealSense SDK 2.0\\lib\\x86")
    set(DLL_DIR "C:\\Program Files (x86)\\Intel RealSense SDK 2.0\\bin\\x86")
  endif()
  set(PROJECT_BINARY_OUTPUT_PATH "${PROJECT_BINARY_DIR}/\$\(Configuration\)")
  set(ADDITIONAL_INCLUDE_DIRS "C:\\Program Files (x86)\\Intel RealSense SDK 2.0\\include")
else(WIN32)
  set(LIBRARY_DIR "/usr/local/lib")
endif(WIN32)


find_library(REALSENSE2_FOUND realsense2 HINTS ${LIBRARY_DIR} REQUIRED)
if(NOT REALSENSE2_FOUND)
  SET(REALSENSE2_FOUND "realsense2")
  message(WARN "Failed to find_library(realsense2)")
else(NOT REALSENSE2_FOUND)
  message(LOG "found in ${REALSENSE2_FOUND}")
endif(NOT REALSENSE2_FOUND)


include_directories(
  ${OpenCV_INCLUDE_DIRS}
  ${LIBRARY_DIR}/../include
)

add_executable(rs-rgbd-viewer rs-rgbd-viewer/rs-rgbd-viewer.cpp)
target_compile_options(rs-rgbd-viewer PRIVATE
  $<$<CXX_COMPILER_ID:MSVC>:/W4 /WX>
  $<$<NOT:$<CXX_COMPILER_ID:MSVC>>:-Wall -Wextra -pedantic -Werror -g -ggdb>
)
target_link_libraries(rs-rgbd-viewer ${OpenCV_LIBS} ${REALSENSE2_FOUND})


# install the DLL for windows folks
if(WIN32)
  message(STATUS "Adding Post build script to copy realsense2.dll to project's binary folder")
  message(STATUS "Copying from ${DLL_DIR} to ${PROJECT_BINARY_OUTPUT_PATH}")
  add_custom_command(TARGET rs-rgbd-viewer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${DLL_DIR}/realsense2.dll"
    ${PROJECT_BINARY_OUTPUT_PATH})
  add_custom_command(TARGET rs-rgbd-viewer POST_BUILD
    COMMAND ${CMAKE_COMMAND} -E copy_if_different
    "${OpenCV_LIBS}"
    ${PROJECT_BINARY_OUTPUT_PATH})
endif(WIN32)

install(TARGETS rs-rgbd-viewer)
