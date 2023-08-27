cmake_minimum_required(VERSION 3.25)
project(nBody LANGUAGES CUDA CXX)

include_directories(src)

find_package(CUDA REQUIRED)
find_package(glfw3 3.3 REQUIRED)
find_package(Vulkan REQUIRED)

add_executable(${PROJECT_NAME} src/main.cpp 
                               src/renderer.cpp 
                               src/nBody.cu)

target_compile_options(nBody PRIVATE -Wall -Wextra -O3)

target_include_directories(${PROJECT_NAME} PRIVATE ${CUDA_INCLUDE_DIRS})
target_link_libraries(${PROJECT_NAME} PRIVATE glfw)
target_link_libraries(${PROJECT_NAME} PRIVATE Vulkan::Vulkan)

find_library(SHADERC_LIB_RELEASE shaderc_combined $ENV{VULKAN_SDK}/Lib)
target_link_libraries(${PROJECT_NAME} PRIVATE ${SHADERC_LIB_RELEASE})

