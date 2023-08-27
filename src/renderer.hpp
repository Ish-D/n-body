#define GLFW_INCLUDE_VULKAN
#include <GLFW/glfw3.h>

#define VULKAN_HPP_NO_CONSTRUCTORS
#include <shaderc/shaderc.hpp>
#include <vulkan/vulkan.hpp>

#include "linmath.h"
#include "nBody.h"

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#include <algorithm>
#include <exception>
#include <limits>
#include <set>
#include <string>
#include <vector>

class Renderer {
  public:
    Renderer(nBody _sim);
    auto init() -> void;
    auto mainLoop() -> void;

    // Utility functions
    auto findSupportedFormat(const std::vector<vk::Format> &candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features) -> vk::Format;
    auto createImage(uint32_t width, uint32_t height, uint32_t mips, vk::Format format, vk::ImageTiling tiling, vk::ImageUsageFlags usage,
                     vk::MemoryPropertyFlags properties, vk::Image &image, vk::DeviceMemory &imageMemory) -> void;
    constexpr auto findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) -> uint32_t;
    auto createBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Buffer &buffer,
                      vk::DeviceMemory &bufferMemory) -> void;
    auto copyBuffer(vk::Buffer dst, vk::Buffer src, vk::DeviceSize size) -> void;

    auto getMemHandle(vk::DeviceMemory memory, vk::ExternalMemoryHandleTypeFlagBits handleType) -> void *;
    auto importExternalSemaphore(cudaExternalSemaphore_t &cudaSem, vk::Semaphore &vkSem, vk::ExternalSemaphoreHandleTypeFlagBits handleType) -> void;
    auto createExternalSemaphore(vk::Semaphore &semaphore, vk::ExternalSemaphoreHandleTypeFlagBits handleType) -> void;
    auto createExternalBuffer(vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties,
                              vk::ExternalMemoryHandleTypeFlagsKHR extMemHandleType, vk::Buffer &buffer, vk::DeviceMemory &bufferMemory) -> void;
    auto importExternalMemory(void **cudaPtr, cudaExternalMemory_t &cudaMem, vk::DeviceMemory &vkMem, vk::DeviceSize size,
                              vk::ExternalMemoryHandleTypeFlagBits handleType) -> void;

    auto fillRenderingCommandBuffer(vk::CommandBuffer &commandBuffer) -> void;
    auto beginSingleTimeCommands() -> std::vector<vk::CommandBuffer>;
    auto endSingleTimeCommands(std::vector<vk::CommandBuffer> commandBuffer) -> void;

  private:
    static constexpr int width                   = 1280;
    static constexpr int height                  = 800;
    static constexpr size_t MAX_FRAMES_IN_FLIGHT = 3;
    const std::string appName                    = "nBody";
    nBody sim;

    struct UniformBufferObject {
        mat4x4 modelViewProj;
    } ubo;

    size_t currentFrame;
    size_t lastFrame;
    float lastTime;

    const std::vector<const char *> validationLayers = {"VK_LAYER_KHRONOS_validation", "VK_LAYER_LUNARG_monitor"};
    const std::vector<const char *> deviceExtensions = {VK_KHR_SWAPCHAIN_EXTENSION_NAME,          VK_KHR_DYNAMIC_RENDERING_EXTENSION_NAME,
                                                        VK_KHR_EXTERNAL_SEMAPHORE_EXTENSION_NAME, VK_KHR_TIMELINE_SEMAPHORE_EXTENSION_NAME,
                                                        VK_KHR_EXTERNAL_MEMORY_FD_EXTENSION_NAME, VK_KHR_EXTERNAL_SEMAPHORE_FD_EXTENSION_NAME,
                                                        VK_KHR_EXTERNAL_MEMORY_EXTENSION_NAME};
    vk::DebugUtilsMessengerEXT messenger;

    GLFWwindow *window;
    vk::Instance instance;
    vk::SurfaceKHR surface;
    vk::PhysicalDevice physicalDevice;
    vk::Device device;
    vk::Queue graphicsQueue;
    vk::Queue presentQueue;

    vk::SwapchainKHR swapChain;
    vk::SurfaceFormatKHR swapChainFormat;
    vk::Extent2D swapChainExtent;
    std::vector<vk::Image> swapChainImages;
    std::vector<vk::ImageView> swapChainImageViews;
    bool windowResize = false;

    vk::PipelineLayout pipelineLayout;
    vk::Pipeline pipeline;

    vk::Image depthImage;
    vk::DeviceMemory depthMemory;
    vk::ImageView depthImageView;

    vk::CommandPool commandPool;
    std::vector<vk::CommandBuffer> commandBuffers;

    std::vector<vk::Buffer> uniformBuffers;
    std::vector<vk::DeviceMemory> uniformBuffersMemory;
    std::vector<void *> uniformBuffersMapped;

    vk::DescriptorPool descriptorPool;
    std::vector<vk::DescriptorSet> descriptorSets;
    vk::DescriptorSetLayout descriptorSetLayout;

    // Sync objects
    std::vector<vk::Fence> inFlightFences;
    std::vector<vk::Semaphore> imageAvailableSemaphores;
    std::vector<vk::Semaphore> renderFinishedSemaphore;
    vk::Semaphore presentationSemaphore;
    vk::Semaphore timelineSemaphore;
    uint8_t deviceUUID[VK_UUID_SIZE];

    // CUDA objects
    cudaExternalSemaphore_t cudaWaitSemaphore, cudaSignalSemaphore, cudaTimelineSemaphore;
    cudaStream_t stream;
    vk::Buffer indexBuffer, vertexBuffer;
    vk::DeviceMemory indexMemory, vertexMemory;
    cudaExternalMemory_t cudaVertMem;

    nBody::point *points;
    size_t numPoints;
    std::vector<uint16_t> indices;

    auto initWindow() -> void;
    auto createInstance() -> void;
    auto createSurface() -> void;
    auto createDevice() -> void;
    auto createSwapChain() -> void;
    auto createImageViews() -> void;
    auto createDescriptorSetLayout() -> void;
    auto createPipeline() -> void;
    auto createFramebuffers() -> void;

    auto initInterop() -> void;
    auto initSimulation() -> void;

    auto createCommandPool() -> void;
    auto createDepthResources() -> void;
    auto createUniformBuffers() -> void;
    auto createDescriptorPool() -> void;
    auto createDescriptorSets() -> void;
    auto createCommandBuffers() -> void;
    auto createSyncObjects() -> void;

    auto drawFrame() -> void;
    auto updateSync() -> void;
    auto updateUniformBuffer(uint32_t imageIndex) -> void;

    auto cleanupSwapChain() -> void;
    auto recreateSwapChain() -> void;
};