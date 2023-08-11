CMAKE_MINIMUM_REQUIRED(VERSION 3.20)



PROJECT(gmc VERSION 1.0 LANGUAGES CXX)

# Set C++ Standard
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_ARM "Enable arm" OFF)

# 指定C编译器
if (ENABLE_ARM)
    set(CMAKE_C_COMPILER /opt/linux/x86-arm/aarch64-mix410-linux/bin/aarch64-mix410-linux-gcc)
    # 指定C++编译器
    set(CMAKE_CXX_COMPILER /opt/linux/x86-arm/aarch64-mix410-linux/bin/aarch64-mix410-linux-g++)
    message("编译arm版本")
endif()

# Set Build Type if not set
if(NOT CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "Choose the type of build." FORCE)
    set_property(CACHE CMAKE_BUILD_TYPE PROPERTY STRINGS "Debug" "Release"
        "MinSizeRel" "RelWithDebInfo")
endif()

# Include directories
include_directories("${PROJECT_SOURCE_DIR}/include")
message("${PROJECT_SOURCE_DIR}/include")



# Collect all source files
file(GLOB_RECURSE SOURCES "${PROJECT_SOURCE_DIR}/src/*.cpp")


add_library(SparseOptFlow_GMC SHARED "${PROJECT_SOURCE_DIR}/src/SparseOptFlow_GMC.cpp")

set_target_properties(SparseOptFlow_GMC PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${PROJECT_SOURCE_DIR}/lib
)


# Create executable
add_executable(${PROJECT_NAME} ${SOURCES})


# Find and link OpenCV
# find_package(OpenCV 4.5 REQUIRED)
# include_directories(${OpenCV_INCLUDE_DIRS})
# target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

# include_directories(/usr/local/opencv4.5.5/include/opencv4)
# link_directories(/usr/local/opencv4.5.5/lib)

if (ENABLE_ARM)
    set(OpenCV_LIBS_PATH ${PROJECT_SOURCE_DIR}/lib/arm)
else()
    set(OpenCV_LIBS_PATH ${PROJECT_SOURCE_DIR}/lib/x86)
endif()

# set(OpenCV_LIBS_PATH /home/huangzm/local/arm-opencv/opencv4.5.5/lib)

file(GLOB OpenCV_LIBS "${OpenCV_LIBS_PATH}/lib*.so")
# set(OpenCV_LIBS libopencv_core.so)
target_link_libraries(${PROJECT_NAME} ${OpenCV_LIBS})

message("${OpenCV_LIBS}")
# message("${OpenCV_INCLUDE_DIRS}")

if(CMAKE_BUILD_TYPE MATCHES Debug)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -pg")
    set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} -pg")
    set(CMAKE_SHARED_LINKER_FLAGS "${CMAKE_SHARED_LINKER_FLAGS} -pg")
    set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -O0")
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O0")
endif()