cmake_minimum_required(VERSION 2.8)
project(test_cpp)

list(APPEND CMAKE_PREFIX_PATH
    ${PROJECT_SOURCE_DIR}/../../../external/libtorch
    /usr/local/cuda
    )
add_definitions(-O3 )

set(CUDA_TOOLKIT_ROOT_DIR "/usr/local/cuda")
find_package(Torch REQUIRED)


include_directories(
        .
        ${TORCH_INCLUDE_DIRS}
)
message("found torch library path: " "${TORCH_LIBRARIES}")


#set(INCLUDE_DIR
#    ${TORCH_INCLUDE_DIRS}
#)

add_executable(test_cpp src/test_cpp.cpp)
target_link_libraries(test_cpp ${TORCH_LIBRARIES})
set_property(TARGET test_cpp PROPERTY CXX_STANDARD 14)
