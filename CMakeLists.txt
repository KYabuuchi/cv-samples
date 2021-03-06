cmake_minimum_required(VERSION 3.1)
project(cv-samples)

# Set default build type to Release
if(NOT DEFINED CMAKE_BUILD_TYPE)
    set(CMAKE_BUILD_TYPE Release)
endif()

# OpenCV
find_package(OpenCV REQUIRED)

# コンパイラーがC++17に対応しているかチェック
include(CheckCXXCompilerFlag)
message("Using C++17")
enable_language(CXX)

# set(CMAKE_CXX_STANDARD 17)
check_cxx_compiler_flag("-std=gnu++17" COMPILER_SUPPORTS_CXX17)
if(NOT ${COMPILER_SUPPORTS_CXX17})
    message(
        FATAL_ERROR
            "${CMAKE_CXX_COMPILER} はC++17をサポートしてません。C++17に対応したコンパイラを指定してください。\n"
        )
endif()
set(CMAKE_CXX_STANDARD 17) # C++14...
set(CMAKE_CXX_STANDARD_REQUIRED ON) # ...is required...
set(CMAKE_CXX_EXTENSIONS OFF) # ...without compiler extensions like gnu++11
message("Compiler:\n\t${CMAKE_CXX_COMPILER}")

# Set warning flags
set(CXX_WARNING_FLAGS
    -Wall
    -Wextra
    -Wconversion
    -Wswitch-default
    -Wdisabled-optimization
    -Wformat
    -Winit-self
    -Woverloaded-virtual
    -Wfloat-equal
    -Wno-old-style-cast
    -Wno-pragmas)
foreach(FLAG IN LISTS CXX_WARNING_FLAGS)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${FLAG}")
endforeach()
message("Build type:\n\t${CMAKE_BUILD_TYPE}")

# epiline_matching
include_directories(${CMAKE_CURRENT_LIST_DIR}/include)
add_executable(epiline epiline_matching.cpp)
target_link_libraries(epiline ${OpenCV_LIBS})
