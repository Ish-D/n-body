cmake_minimum_required(VERSION 3.25)
project(nBody LANGUAGES CUDA CXX)

include_directories(src)

find_package(CUDA REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(Vulkan REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp 
                               src/renderer.cpp 
                               src/nBody.cu)


target_link_libraries(${PROJECT_NAME} Vulkan::Vulkan)
target_link_libraries(${PROJECT_NAME} glfw)
target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})