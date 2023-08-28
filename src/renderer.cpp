#include "renderer.hpp"

Renderer::Renderer(nBody _sim) : sim(_sim) {}

auto Renderer::init() -> void {
    initWindow();
    createInstance();
    createSurface();
    createDevice();
    createSwapChain();
    createImageViews();
    createDescriptorSetLayout();
    createPipeline();
    createCommandPool();
    createDepthResources();

    initInterop();

    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
    createSyncObjects();
}

auto Renderer::initWindow() -> void {
    glfwInit();
    glfwWindowHint(GLFW_CLIENT_API, GLFW_NO_API);

    window = glfwCreateWindow(width, height, appName.c_str(), nullptr, nullptr);

    glfwMakeContextCurrent(window);
}

#ifndef NDEBUG
static VKAPI_ATTR VkBool32 VKAPI_CALL debugCallback([[maybe_unused]] VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
                                                    [[maybe_unused]] VkDebugUtilsMessageTypeFlagsEXT messageType,
                                                    [[maybe_unused]] const VkDebugUtilsMessengerCallbackDataEXT *pCallbackData,
                                                    [[maybe_unused]] void *pUserData) {
    std::cerr << "validation layer: " << pCallbackData->pMessage << std::endl;
    return VK_FALSE;
}
#endif

auto Renderer::createInstance() -> void {
    vk::ApplicationInfo appInfo{.pApplicationName   = appName.c_str(),
                                .applicationVersion = 1,
                                .pEngineName        = appName.c_str(),
                                .engineVersion      = 1,
                                .apiVersion         = VK_API_VERSION_1_3};

    auto glfwExtensionCount     = 0u;
    auto glfwRequiredExtensions = glfwGetRequiredInstanceExtensions(&glfwExtensionCount);
    std::vector<const char *> glfwExtensions(glfwRequiredExtensions, glfwRequiredExtensions + glfwExtensionCount);
    glfwExtensions.push_back("VK_EXT_debug_utils");

    instance = vk::createInstance(vk::InstanceCreateInfo{.pApplicationInfo        = &appInfo,
                                                         .enabledLayerCount       = static_cast<uint32_t>(validationLayers.size()),
                                                         .ppEnabledLayerNames     = validationLayers.data(),
                                                         .enabledExtensionCount   = static_cast<uint32_t>(glfwExtensions.size()),
                                                         .ppEnabledExtensionNames = glfwExtensions.data()});

// enable validation while in debug
#ifndef NDEBUG
    auto dispatch_loader = vk::DispatchLoaderDynamic(instance, vkGetInstanceProcAddr);

    auto debugMessenger = vk::DebugUtilsMessengerCreateInfoEXT{
        .messageSeverity = vk::DebugUtilsMessageSeverityFlagBitsEXT::eError | vk::DebugUtilsMessageSeverityFlagBitsEXT::eWarning |
                           vk::DebugUtilsMessageSeverityFlagBitsEXT::eVerbose | vk::DebugUtilsMessageSeverityFlagBitsEXT::eInfo,
        .messageType = vk::DebugUtilsMessageTypeFlagBitsEXT::eGeneral | vk::DebugUtilsMessageTypeFlagBitsEXT::eValidation |
                       vk::DebugUtilsMessageTypeFlagBitsEXT::ePerformance,
        .pfnUserCallback = debugCallback};
    messenger = instance.createDebugUtilsMessengerEXT(debugMessenger, nullptr, dispatch_loader);
#endif
}

auto Renderer::createSurface() -> void {
    VkSurfaceKHR surfaceTemp;
    if (glfwCreateWindowSurface(instance, window, nullptr, &surfaceTemp) != VK_SUCCESS)
        throw std::runtime_error("Failed to create surface");

    surface = vk::SurfaceKHR(surfaceTemp);
}

static auto findGraphicsQueueIndicies(vk::PhysicalDevice physicalDevice, vk::SurfaceKHR surface, uint32_t &graphicsFamily, uint32_t &presentFamily)
    -> bool {
    const auto queueFamilyProperties = physicalDevice.getQueueFamilyProperties();

    graphicsFamily = presentFamily = ~0;
    for (uint32_t i = 0; i < queueFamilyProperties.size(); i++) {
        if (queueFamilyProperties[i].queueCount > 0u) {
            if (graphicsFamily == ~0u && queueFamilyProperties[i].queueFlags & vk::QueueFlagBits::eGraphics)
                graphicsFamily = i;
        }

        if (presentFamily == ~0u && physicalDevice.getSurfaceSupportKHR(i, surface)) {
            presentFamily = i;
        }

        if (presentFamily != ~0u && graphicsFamily != ~0u)
            break;
    }

    return graphicsFamily != ~0u && presentFamily != ~0u;
}

auto Renderer::createDevice() -> void {
    const auto physicalDevices = instance.enumeratePhysicalDevices();
    if (physicalDevices.size() == 0)
        throw std::runtime_error("Failed to find Vulkan compatible device");

    // lambda returns bool of whether or not a particular device is suitable (desired qualities in bottom)
    auto suitableDevice = [&](const vk::PhysicalDevice &physicalDevice) {
        const auto availableExtensions = physicalDevice.enumerateDeviceExtensionProperties();
        std::set<std::string> extensionSet(deviceExtensions.begin(), deviceExtensions.end());
        for (const auto &[name, version] : availableExtensions)
            extensionSet.erase(name);

        // const auto supportedFeatures = physicalDevice.getFeatures();
        return extensionSet.empty(); // && sampler features for anything else needed
    };

    physicalDevice =
        physicalDevices[std::distance(begin(physicalDevices), std::find_if(begin(physicalDevices), end(physicalDevices), suitableDevice))];

    uint32_t graphicsQueueIndex, presentQueueIndex;
    findGraphicsQueueIndicies(physicalDevice, surface, graphicsQueueIndex, presentQueueIndex);

    std::set<uint32_t> uniqueFamilyIndices = {graphicsQueueIndex, presentQueueIndex};
    float queuePriority                    = 1.0f;
    std::vector<vk::DeviceQueueCreateInfo> queueCreateInfos;
    for (auto &queueFamilyIndex : uniqueFamilyIndices) {
        queueCreateInfos.push_back(vk::DeviceQueueCreateInfo{.flags            = vk::DeviceQueueCreateFlags(),
                                                             .queueFamilyIndex = static_cast<uint32_t>(queueFamilyIndex),
                                                             .queueCount       = 1,
                                                             .pQueuePriorities = &queuePriority});
    }

    // Enable opt-in features
    constexpr auto dynamicRenderingFeatures   = vk::PhysicalDeviceDynamicRenderingFeatures{.dynamicRendering = true};
    constexpr auto timelineSemaphoreFeatures  = vk::PhysicalDeviceTimelineSemaphoreFeatures{.timelineSemaphore = true};
    constexpr auto descriptorIndexingFeatures = vk::PhysicalDeviceDescriptorIndexingFeatures{.shaderSampledImageArrayNonUniformIndexing = true,
                                                                                             .descriptorBindingPartiallyBound           = true,
                                                                                             .descriptorBindingVariableDescriptorCount  = true,
                                                                                             .runtimeDescriptorArray                    = true};

    const auto deviceCreateInfo = vk::DeviceCreateInfo{.flags                   = vk::DeviceCreateFlags(),
                                                       .queueCreateInfoCount    = static_cast<uint32_t>(queueCreateInfos.size()),
                                                       .pQueueCreateInfos       = queueCreateInfos.data(),
                                                       .enabledExtensionCount   = static_cast<uint32_t>(deviceExtensions.size()),
                                                       .ppEnabledExtensionNames = deviceExtensions.data()};

    const vk::StructureChain<vk::DeviceCreateInfo,
                             vk::PhysicalDeviceDynamicRenderingFeatures,
                             vk::PhysicalDeviceTimelineSemaphoreFeatures,
                             vk::PhysicalDeviceDescriptorIndexingFeatures>
        chain = {deviceCreateInfo, dynamicRenderingFeatures, timelineSemaphoreFeatures, descriptorIndexingFeatures};

    device = physicalDevice.createDevice(chain.get<vk::DeviceCreateInfo>());

    graphicsQueue = device.getQueue(graphicsQueueIndex, 0);
    presentQueue  = device.getQueue(presentQueueIndex, 0);

    auto physicalDeviceIDProperties = vk::PhysicalDeviceIDProperties{};
    auto physicalDeviceProperties2  = vk::PhysicalDeviceProperties2{.pNext = &physicalDeviceIDProperties};

    auto fpGetPhysicalDeviceProperties2 = (PFN_vkGetPhysicalDeviceProperties2)vkGetInstanceProcAddr(instance, "vkGetPhysicalDeviceProperties2");
    if (fpGetPhysicalDeviceProperties2 == nullptr)
        throw std::runtime_error("Proc Address for vkGetPhysicalDeviceProperties2 not found");

    fpGetPhysicalDeviceProperties2(physicalDevice, reinterpret_cast<VkPhysicalDeviceProperties2 *>(&physicalDeviceProperties2));
    memcpy(deviceUUID, physicalDeviceIDProperties.deviceUUID, VK_UUID_SIZE);
}

auto Renderer::createSwapChain() -> void {

    // lambdas to help with choosing swap chain properties
    auto chooseSwapSurfaceFormat = [&](const std::vector<vk::SurfaceFormatKHR> formats) -> vk::SurfaceFormatKHR {
        if (formats.size() == 1 && formats[0].format == vk::Format::eUndefined)
            return {vk::Format::eB8G8R8A8Unorm, vk::ColorSpaceKHR::eSrgbNonlinear};
        for (const auto &availableFormat : formats) {
            if (availableFormat.format == vk::Format::eB8G8R8A8Unorm && availableFormat.colorSpace == vk::ColorSpaceKHR::eSrgbNonlinear)
                return availableFormat;
        }

        return formats[0];
    };

    auto chooseSwapPresentMode = [&](std::vector<vk::PresentModeKHR> presentModes) -> vk::PresentModeKHR {
        vk::PresentModeKHR bestMode = vk::PresentModeKHR::eFifo;

        for (const auto &availablePresentMode : presentModes) {
            if (availablePresentMode == vk::PresentModeKHR::eMailbox)
                return availablePresentMode;
            else if (availablePresentMode == vk::PresentModeKHR::eImmediate)
                bestMode = availablePresentMode;
        }

        return bestMode;
    };

    auto chooseSwapExtent = [&](GLFWwindow *window, vk::SurfaceCapabilitiesKHR capabilities) -> vk::Extent2D {
        if (capabilities.currentExtent.width != std::numeric_limits<uint32_t>::max())
            return capabilities.currentExtent;
        else {
            int width, height;
            glfwGetFramebufferSize(window, &width, &height);
            vk::Extent2D actual = {static_cast<uint32_t>(width), static_cast<uint32_t>(height)};
            actual.width        = std::max(capabilities.minImageExtent.width, std::min(capabilities.maxImageExtent.width, actual.width));
            actual.height       = std::max(capabilities.minImageExtent.height, std::min(capabilities.maxImageExtent.height, actual.height));

            return actual;
        }
    };

    const auto capabilities = physicalDevice.getSurfaceCapabilitiesKHR(surface);
    const auto format       = chooseSwapSurfaceFormat(physicalDevice.getSurfaceFormatsKHR(surface));
    const auto presentMode  = chooseSwapPresentMode(physicalDevice.getSurfacePresentModesKHR(surface));
    const auto extent       = chooseSwapExtent(window, capabilities);

    uint32_t imageCount = capabilities.minImageCount + 1;
    if (capabilities.maxImageCount > 0 && imageCount > capabilities.maxImageCount)
        imageCount = capabilities.maxImageCount;

    vk::SwapchainCreateInfoKHR swapChainCreateInfo{.surface          = surface,
                                                   .minImageCount    = imageCount,
                                                   .imageFormat      = format.format,
                                                   .imageColorSpace  = vk::ColorSpaceKHR::eSrgbNonlinear,
                                                   .imageExtent      = extent,
                                                   .imageArrayLayers = 1,
                                                   .imageUsage       = vk::ImageUsageFlagBits::eColorAttachment,
                                                   .preTransform     = capabilities.currentTransform,
                                                   .compositeAlpha   = vk::CompositeAlphaFlagBitsKHR::eOpaque,
                                                   .presentMode      = presentMode,
                                                   .clipped          = true,
                                                   .oldSwapchain     = nullptr};
    std::array<uint32_t, 2> queueFamilyIndices;
    findGraphicsQueueIndicies(physicalDevice, surface, queueFamilyIndices[0], queueFamilyIndices[1]);
    if (queueFamilyIndices[0] != queueFamilyIndices[1]) {
        swapChainCreateInfo.imageSharingMode      = vk::SharingMode::eConcurrent;
        swapChainCreateInfo.queueFamilyIndexCount = queueFamilyIndices.size();
        swapChainCreateInfo.pQueueFamilyIndices   = queueFamilyIndices.data();
    } else {
        swapChainCreateInfo.imageSharingMode = vk::SharingMode::eExclusive;
    }

    swapChain       = device.createSwapchainKHR(swapChainCreateInfo);
    swapChainImages = device.getSwapchainImagesKHR(swapChain);

    swapChainFormat = format;
    swapChainExtent = extent;
}

auto Renderer::createImageViews() -> void {
    swapChainImageViews.reserve(swapChainImages.size());
    for (const auto &image : swapChainImages) {
        vk::ImageViewCreateInfo imageViewCreateInfo{
            .flags            = vk::ImageViewCreateFlags(),
            .image            = image,
            .viewType         = vk::ImageViewType::e2D,
            .format           = swapChainFormat.format,
            .components       = vk::ComponentMapping{vk::ComponentSwizzle::eIdentity, vk::ComponentSwizzle::eIdentity},
            .subresourceRange = vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1}
        };
        swapChainImageViews.push_back(device.createImageView(imageViewCreateInfo));
    }
}

auto Renderer::createDescriptorSetLayout() -> void {
    const auto uboLayoutBinding = vk::DescriptorSetLayoutBinding{
        .binding = 0, .descriptorType = vk::DescriptorType::eUniformBuffer, .descriptorCount = 1, .stageFlags = vk::ShaderStageFlagBits::eVertex};
    const auto samplerLayoutBinding = vk::DescriptorSetLayoutBinding{.binding         = 1,
                                                                     .descriptorType  = vk::DescriptorType::eCombinedImageSampler,
                                                                     .descriptorCount = 1,
                                                                     .stageFlags      = vk::ShaderStageFlagBits::eFragment};

    std::array<vk::DescriptorSetLayoutBinding, 2> bindings{uboLayoutBinding, samplerLayoutBinding};
    descriptorSetLayout = device.createDescriptorSetLayout(
        vk::DescriptorSetLayoutCreateInfo{.bindingCount = static_cast<uint32_t>(bindings.size()), .pBindings = bindings.data()});
}

static std::string readFile(const std::string filename) {
    const std::string path = "../src/shaders/" + filename;
    const std::ifstream file(path);

    if (!file.is_open())
        throw std::runtime_error("Failed to open shader file");

    std::stringstream buffer;
    buffer << file.rdbuf();
    return buffer.str();
}

auto getVertexDescriptions(std::vector<vk::VertexInputBindingDescription> &vertexBindingDescriptions,
                           std::vector<vk::VertexInputAttributeDescription> &vertexAttributeDescriptions) -> void {
    vertexBindingDescriptions.resize(1);
    vertexAttributeDescriptions.resize(3);

    // x,y,z,size
    vertexBindingDescriptions[0].binding   = 0;
    vertexBindingDescriptions[0].stride    = sizeof(Point);
    vertexBindingDescriptions[0].inputRate = vk::VertexInputRate::eVertex;

    vertexAttributeDescriptions[0].binding  = 0;
    vertexAttributeDescriptions[0].location = 0;
    vertexAttributeDescriptions[0].format   = vk::Format::eR32G32B32Sfloat;
    vertexAttributeDescriptions[0].offset   = 0;

    // color
    vertexAttributeDescriptions[1].binding  = 0;
    vertexAttributeDescriptions[1].location = 1;
    vertexAttributeDescriptions[1].format   = vk::Format::eR32G32B32Sfloat;
    vertexAttributeDescriptions[1].offset   = 3 * sizeof(float);

    // size
    vertexAttributeDescriptions[2].binding  = 0;
    vertexAttributeDescriptions[2].location = 2;
    vertexAttributeDescriptions[2].format   = vk::Format::eR32Sfloat;
    vertexAttributeDescriptions[2].offset   = 6 * sizeof(float);
}

auto Renderer::createPipeline() -> void {
    // read in both shaders
    const std::string vertShader = readFile("shader.vert");
    const std::string fragShader = readFile("shader.frag");

    shaderc::Compiler compiler;
    shaderc::CompileOptions options;
    options.SetOptimizationLevel(shaderc_optimization_level_performance);

    // compile shaders with shaderc
    shaderc::SpvCompilationResult vertShaderModule = compiler.CompileGlslToSpv(vertShader, shaderc_glsl_vertex_shader, "vertex shader", options);
    if (vertShaderModule.GetCompilationStatus() != shaderc_compilation_status_success)
        std::cerr << vertShaderModule.GetErrorMessage();
    auto vertShaderCode       = std::vector<uint32_t>{vertShaderModule.cbegin(), vertShaderModule.cend()};
    auto vertSize             = std::distance(vertShaderCode.begin(), vertShaderCode.end());
    auto vertShaderCreateInfo = vk::ShaderModuleCreateInfo{.codeSize = vertSize * sizeof(uint32_t), .pCode = vertShaderCode.data()};
    auto vertexShaderModule   = device.createShaderModule(vertShaderCreateInfo);

    shaderc::SpvCompilationResult fragShaderModule = compiler.CompileGlslToSpv(fragShader, shaderc_glsl_fragment_shader, "fragment shader", options);
    if (fragShaderModule.GetCompilationStatus() != shaderc_compilation_status_success)
        std::cerr << fragShaderModule.GetErrorMessage();
    auto fragShaderCode       = std::vector<uint32_t>{fragShaderModule.cbegin(), fragShaderModule.cend()};
    auto fragSize             = std::distance(fragShaderCode.begin(), fragShaderCode.end());
    auto fragShaderCreateInfo = vk::ShaderModuleCreateInfo{.codeSize = fragSize * sizeof(uint32_t), .pCode = fragShaderCode.data()};
    auto fragmentShaderModule = device.createShaderModule(fragShaderCreateInfo);

    auto vertShaderStageInfo =
        vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eVertex, .module = vertexShaderModule, .pName = "main"};

    auto fragShaderStageInfo =
        vk::PipelineShaderStageCreateInfo{.stage = vk::ShaderStageFlagBits::eFragment, .module = fragmentShaderModule, .pName = "main"};

    auto pipelineShaderStages = std::vector<vk::PipelineShaderStageCreateInfo>{vertShaderStageInfo, fragShaderStageInfo};

    std::vector<vk::VertexInputBindingDescription> vertexBindingDescriptions;
    std::vector<vk::VertexInputAttributeDescription> vertexAttributeDescriptions;

    getVertexDescriptions(vertexBindingDescriptions, vertexAttributeDescriptions);

    auto vertexInputInfo =
        vk::PipelineVertexInputStateCreateInfo{.vertexBindingDescriptionCount   = static_cast<uint32_t>(vertexBindingDescriptions.size()),
                                               .pVertexBindingDescriptions      = vertexBindingDescriptions.data(),
                                               .vertexAttributeDescriptionCount = static_cast<uint32_t>(vertexAttributeDescriptions.size()),
                                               .pVertexAttributeDescriptions    = vertexAttributeDescriptions.data()};

    auto inputAssembly = vk::PipelineInputAssemblyStateCreateInfo{.topology = vk::PrimitiveTopology::ePointList, .primitiveRestartEnable = false};

    auto viewport = vk::Viewport{0.0f, 0.0f, static_cast<float>(swapChainExtent.width), static_cast<float>(swapChainExtent.height), 0.0f, 1.0f};
    auto scissor  = vk::Rect2D{
         {0, 0},
         swapChainExtent
    };

    auto viewportState = vk::PipelineViewportStateCreateInfo{.viewportCount = 1, .pViewports = &viewport, .scissorCount = 1, .pScissors = &scissor};

    auto rasterizer = vk::PipelineRasterizationStateCreateInfo{.depthClampEnable        = false,
                                                               .rasterizerDiscardEnable = false,
                                                               .polygonMode             = vk::PolygonMode::eFill,
                                                               .cullMode                = vk::CullModeFlagBits::eNone,
                                                               .frontFace               = vk::FrontFace::eClockwise,
                                                               .depthBiasEnable         = false,
                                                               .lineWidth               = 1.0f};

    auto multisampling = vk::PipelineMultisampleStateCreateInfo{
        .rasterizationSamples = vk::SampleCountFlagBits::e1, .sampleShadingEnable = false, .minSampleShading = 1.0};

    auto colorBlendAttachment =
        vk::PipelineColorBlendAttachmentState{.blendEnable         = true,
                                              .srcColorBlendFactor = vk::BlendFactor::eSrcAlpha,
                                              .dstColorBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
                                              .colorBlendOp        = vk::BlendOp::eAdd,
                                              .srcAlphaBlendFactor = vk::BlendFactor::eSrcAlpha,
                                              .dstAlphaBlendFactor = vk::BlendFactor::eOneMinusSrcAlpha,
                                              .alphaBlendOp        = vk::BlendOp::eSubtract,
                                              .colorWriteMask      = vk::ColorComponentFlagBits::eR | vk::ColorComponentFlagBits::eG |
                                                                vk::ColorComponentFlagBits::eB | vk::ColorComponentFlagBits::eA};

    auto colorblending = vk::PipelineColorBlendStateCreateInfo{.logicOpEnable   = false,
                                                               .logicOp         = vk::LogicOp::eCopy,
                                                               .attachmentCount = 1,
                                                               .pAttachments    = &colorBlendAttachment,
                                                               .blendConstants  = vk::ArrayWrapper1D<float, 4>()};

    auto depthStencil = vk::PipelineDepthStencilStateCreateInfo{.depthTestEnable       = true,
                                                                .depthWriteEnable      = true,
                                                                .depthCompareOp        = vk::CompareOp::eLess,
                                                                .depthBoundsTestEnable = false,
                                                                .stencilTestEnable     = false,
                                                                .minDepthBounds        = 0.0f,
                                                                .maxDepthBounds        = 0.0f};

    pipelineLayout = device.createPipelineLayout(vk::PipelineLayoutCreateInfo{.setLayoutCount = 1, .pSetLayouts = &descriptorSetLayout}, nullptr);

    auto depthFormat = findSupportedFormat(
        {vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint}, vk::ImageTiling::eOptimal, vk::FormatFeatureFlagBits::eDepthStencilAttachment);

    auto pipelineRenderingInfo = vk::PipelineRenderingCreateInfo{.colorAttachmentCount    = 1,
                                                                 .pColorAttachmentFormats = &swapChainFormat.format,
                                                                 .depthAttachmentFormat   = depthFormat,
                                                                 .stencilAttachmentFormat = depthFormat};

    auto pipelineCreateInfo = vk::GraphicsPipelineCreateInfo{.pNext               = &pipelineRenderingInfo,
                                                             .stageCount          = 2,
                                                             .pStages             = pipelineShaderStages.data(),
                                                             .pVertexInputState   = &vertexInputInfo,
                                                             .pInputAssemblyState = &inputAssembly,
                                                             .pTessellationState  = nullptr,
                                                             .pViewportState      = &viewportState,
                                                             .pRasterizationState = &rasterizer,
                                                             .pMultisampleState   = &multisampling,
                                                             .pDepthStencilState  = &depthStencil,
                                                             .pColorBlendState    = &colorblending,
                                                             .pDynamicState       = nullptr,
                                                             .layout              = pipelineLayout,
                                                             .renderPass          = nullptr,
                                                             .subpass             = 0};

    pipeline = device.createGraphicsPipeline({}, pipelineCreateInfo).value;

    device.destroyShaderModule(vertexShaderModule);
    device.destroyShaderModule(fragmentShaderModule);
}

auto Renderer::createCommandPool() -> void {
    uint32_t graphicsQueueFamilyIndex, presentQueueFamilyIndex;
    findGraphicsQueueIndicies(physicalDevice, surface, graphicsQueueFamilyIndex, presentQueueFamilyIndex);

    commandPool = device.createCommandPool(vk::CommandPoolCreateInfo{.queueFamilyIndex = static_cast<uint32_t>(graphicsQueueFamilyIndex)});
}

auto Renderer::createDepthResources() -> void {
    const auto depthFormat = findSupportedFormat(
        {vk::Format::eD32SfloatS8Uint, vk::Format::eD24UnormS8Uint}, vk::ImageTiling::eOptimal, vk::FormatFeatureFlagBits::eDepthStencilAttachment);
    createImage(swapChainExtent.width,
                swapChainExtent.height,
                1,
                depthFormat,
                vk::ImageTiling::eOptimal,
                vk::ImageUsageFlagBits::eDepthStencilAttachment,
                vk::MemoryPropertyFlagBits::eDeviceLocal,
                depthImage,
                depthMemory);
    depthImageView = device.createImageView(vk::ImageViewCreateInfo{
        .image            = depthImage,
        .viewType         = vk::ImageViewType::e2D,
        .format           = depthFormat,
        .subresourceRange = vk::ImageSubresourceRange{vk::ImageAspectFlagBits::eDepth, 0, 1, 0, 1}
    });
}

auto Renderer::createUniformBuffers() -> void {
    uniformBuffers.resize(swapChainImages.size());
    uniformBuffersMapped.resize(swapChainImages.size());
    uniformBuffersMemory.resize(swapChainImages.size());

    for (size_t i = 0; i < uniformBuffers.size(); i++) {
        createBuffer(sizeof(UniformBufferObject),
                     vk::BufferUsageFlagBits::eUniformBuffer,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                     uniformBuffers[i],
                     uniformBuffersMemory[i]);
    }
}

auto Renderer::createDescriptorPool() -> void {
    const auto poolSize =
        vk::DescriptorPoolSize{.type = vk::DescriptorType::eUniformBuffer, .descriptorCount = static_cast<uint32_t>(swapChainImages.size())};

    descriptorPool = device.createDescriptorPool(vk::DescriptorPoolCreateInfo{
        .maxSets       = static_cast<uint32_t>(swapChainImages.size()),
        .poolSizeCount = 1,
        .pPoolSizes    = &poolSize,
    });
}

auto Renderer::createDescriptorSets() -> void {
    std::vector<vk::DescriptorSetLayout> layouts(swapChainImages.size(), descriptorSetLayout);

    const auto allocInfo = vk::DescriptorSetAllocateInfo{
        .descriptorPool = descriptorPool, .descriptorSetCount = static_cast<uint32_t>(swapChainImages.size()), .pSetLayouts = layouts.data()};
    descriptorSets.resize(swapChainImages.size());
    if (device.allocateDescriptorSets(&allocInfo, descriptorSets.data()) != vk::Result::eSuccess)
        throw std::runtime_error("Failed to allocate descriptor sets");

    for (size_t i = 0; i < swapChainImages.size(); i++) {
        auto bufferInfo = vk::DescriptorBufferInfo{.offset = 0, .range = VK_WHOLE_SIZE};

        vk::WriteDescriptorSet descriptorWrite;

        descriptorWrite = vk::WriteDescriptorSet{.dstBinding      = 0,
                                                 .dstArrayElement = 0,
                                                 .descriptorCount = 1,
                                                 .descriptorType  = vk::DescriptorType::eUniformBuffer,
                                                 .pBufferInfo     = &bufferInfo};

        for (size_t i = 0; i < swapChainImages.size(); i++) {
            bufferInfo.buffer      = uniformBuffers[i];
            descriptorWrite.dstSet = descriptorSets[i];
            device.updateDescriptorSets(1, &descriptorWrite, 0, nullptr);
        }
    }
}

auto Renderer::fillRenderingCommandBuffer(vk::CommandBuffer &commandBuffer) -> void {
    vk::Buffer vertexBuffers[] = {vertexBuffer};
    vk::DeviceSize offsets[]   = {0};
    commandBuffer.bindVertexBuffers(0, 1, vertexBuffers, offsets);
    commandBuffer.bindIndexBuffer(indexBuffer, 0, vk::IndexType::eUint16);
    commandBuffer.drawIndexed(static_cast<uint32_t>(indices.size()), 1, 0, 0, 0);
}

auto Renderer::createCommandBuffers() -> void {
    commandBuffers.resize(swapChainImages.size());
    commandBuffers = device.allocateCommandBuffers(vk::CommandBufferAllocateInfo{
        .commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = static_cast<uint32_t>(commandBuffers.size())});

    const vk::ImageSubresourceRange subresourceRange = {vk::ImageAspectFlagBits::eColor, 0, 1, 0, 1};
    const vk::ImageSubresourceRange depthRange       = {vk::ImageAspectFlagBits::eDepth | vk::ImageAspectFlagBits::eStencil, 0, 1, 0, 1};

    const std::array<vk::ClearValue, 2> clearValues{vk::ClearValue{.color = {std::array<float, 4>()}},
                                                    vk::ClearValue{.depthStencil = vk::ClearDepthStencilValue{1.0f, 0}}};

    for (size_t i = 0; i < commandBuffers.size(); i++) {
        commandBuffers[i].begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eSimultaneousUse});

        auto imageMemoryBarrier1 = vk::ImageMemoryBarrier{.srcAccessMask    = vk::AccessFlagBits::eColorAttachmentWrite,
                                                          .oldLayout        = vk::ImageLayout::eUndefined,
                                                          .newLayout        = vk::ImageLayout::eColorAttachmentOptimal,
                                                          .image            = swapChainImages[i],
                                                          .subresourceRange = subresourceRange};
        commandBuffers[i].pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                          vk::PipelineStageFlagBits::eTopOfPipe,
                                          {},
                                          0,
                                          nullptr,
                                          0,
                                          nullptr,
                                          1,
                                          &imageMemoryBarrier1);

        auto depthMemoryBarrier1 = vk::ImageMemoryBarrier{.srcAccessMask    = vk::AccessFlagBits::eDepthStencilAttachmentWrite,
                                                          .oldLayout        = vk::ImageLayout::eUndefined,
                                                          .newLayout        = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                                          .image            = depthImage,
                                                          .subresourceRange = depthRange};
        commandBuffers[i].pipelineBarrier(vk::PipelineStageFlagBits::eLateFragmentTests | vk::PipelineStageFlagBits::eEarlyFragmentTests,
                                          vk::PipelineStageFlagBits::eLateFragmentTests | vk::PipelineStageFlagBits::eEarlyFragmentTests,
                                          {},
                                          0,
                                          nullptr,
                                          0,
                                          nullptr,
                                          1,
                                          &depthMemoryBarrier1);

        auto colorAttachment = vk::RenderingAttachmentInfo{.imageView   = swapChainImageViews[i],
                                                           .imageLayout = vk::ImageLayout::eAttachmentOptimal,
                                                           .loadOp      = vk::AttachmentLoadOp::eClear,
                                                           .storeOp     = vk::AttachmentStoreOp::eStore,
                                                           .clearValue  = clearValues[0]};
        auto depthAttachment = vk::RenderingAttachmentInfo{.imageView   = depthImageView,
                                                           .imageLayout = vk::ImageLayout::eDepthStencilAttachmentOptimal,
                                                           .loadOp      = vk::AttachmentLoadOp::eClear,
                                                           .storeOp     = vk::AttachmentStoreOp::eStore,
                                                           .clearValue  = clearValues[1]};

        auto renderingInfo = vk::RenderingInfo{
            .renderArea           = {{0, 0}, {swapChainExtent.width, swapChainExtent.height}},
            .layerCount           = 1,
            .colorAttachmentCount = 1,
            .pColorAttachments    = &colorAttachment,
            .pDepthAttachment     = &depthAttachment,
            .pStencilAttachment   = &depthAttachment
        };
        commandBuffers[i].beginRendering(&renderingInfo);
        commandBuffers[i].bindPipeline(vk::PipelineBindPoint::eGraphics, pipeline);
        commandBuffers[i].bindDescriptorSets(vk::PipelineBindPoint::eGraphics, pipelineLayout, 0, 1, &descriptorSets[i], 0, nullptr);
        fillRenderingCommandBuffer(commandBuffers[i]);
        commandBuffers[i].endRendering();

        auto imageMemoryBarrier2 = vk::ImageMemoryBarrier{.srcAccessMask    = vk::AccessFlagBits::eColorAttachmentWrite,
                                                          .oldLayout        = vk::ImageLayout::eColorAttachmentOptimal,
                                                          .newLayout        = vk::ImageLayout::ePresentSrcKHR,
                                                          .image            = swapChainImages[i],
                                                          .subresourceRange = subresourceRange};
        commandBuffers[i].pipelineBarrier(vk::PipelineStageFlagBits::eColorAttachmentOutput,
                                          vk::PipelineStageFlagBits::eBottomOfPipe,
                                          {},
                                          0,
                                          nullptr,
                                          0,
                                          nullptr,
                                          1,
                                          &imageMemoryBarrier2);
        commandBuffers[i].end();
    }
}

auto Renderer::createSyncObjects() -> void {
    inFlightFences.resize(MAX_FRAMES_IN_FLIGHT);
    imageAvailableSemaphores.resize(MAX_FRAMES_IN_FLIGHT);
    renderFinishedSemaphore.resize(MAX_FRAMES_IN_FLIGHT);

    for (size_t i = 0; i < MAX_FRAMES_IN_FLIGHT; i++) {
        inFlightFences[i]           = device.createFence(vk::FenceCreateInfo{.flags = vk::FenceCreateFlagBits::eSignaled}, nullptr);
        imageAvailableSemaphores[i] = device.createSemaphore(vk::SemaphoreCreateInfo{}, nullptr);
        renderFinishedSemaphore[i]  = device.createSemaphore(vk::SemaphoreCreateInfo{}, nullptr);
    }

    presentationSemaphore = device.createSemaphore(vk::SemaphoreCreateInfo{}, nullptr);
}

auto Renderer::updateUniformBuffer(uint32_t imageIndex) -> void {
    {
        mat4x4 view, proj;
        vec3 eye    = {1.75f, 1.75f, 1.25f};
        vec3 center = {0.0f, 0.0f, -0.25f};
        vec3 up     = {0.0f, 0.0f, 1.0f};

        mat4x4_perspective(proj, (float)degreesToRadians(45.0f), swapChainExtent.width / (float)swapChainExtent.height, 0.1f, 10.0f);
        proj[1][1] *= -1.0f; // Flip y axis

        mat4x4_look_at(view, eye, center, up);
        mat4x4_mul(ubo.modelViewProj, proj, view);
    }

    void *data;
    if (device.mapMemory(uniformBuffersMemory[imageIndex], 0, sizeof(UniformBufferObject), {}, &data) != vk::Result::eSuccess)
        throw std::runtime_error("Failed to update uniform buffer memory");
    memcpy(data, &ubo, sizeof(ubo));
    device.unmapMemory(uniformBuffersMemory[imageIndex]);
}

auto Renderer::cleanupSwapChain() -> void {
    if (depthImageView)
        device.destroyImageView(depthImageView);
    if (depthImage)
        device.destroyImage(depthImage);
    if (depthMemory)
        device.freeMemory(depthMemory);
    for (size_t i = 0; i < uniformBuffers.size(); i++) {
        device.destroyBuffer(uniformBuffers[i]);
        device.freeMemory(uniformBuffersMemory[i]);
    }

    if (descriptorPool)
        device.destroyDescriptorPool(descriptorPool);
    if (pipeline)
        device.destroyPipeline(pipeline);
    if (pipelineLayout)
        device.destroyPipelineLayout(pipelineLayout);
    for (size_t i = 0; i < swapChainImageViews.size(); i++) {
        device.destroyImageView(swapChainImageViews[i]);
    }

    if (swapChain)
        device.destroySwapchainKHR(swapChain);
}

auto Renderer::recreateSwapChain() -> void {
    int width, height;
    glfwGetFramebufferSize(window, &width, &height);
    while (width == 0 || height == 0) {
        glfwWaitEvents();
        glfwGetFramebufferSize(window, &width, &height);
    }

    cleanupSwapChain();

    device.waitIdle();
    createSwapChain();
    createImageViews();
    createPipeline();
    createDepthResources();
    createUniformBuffers();
    createDescriptorPool();
    createDescriptorSets();
    createCommandBuffers();
}

auto Renderer::initInterop() -> void {
    int cudaDevice = sim.initCuda(deviceUUID, VK_UUID_SIZE);

    if (cudaDevice == -1) {
        std::cout << "No CUDA-Vulkan interop capable device found\n";
        exit(EXIT_FAILURE);
    }

    sim.initCudaLaunchConfig(cudaDevice);
    checkCudaErrors(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));

    createExternalBuffer(sim.getNumPoints() * sizeof(Point),
                         vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eVertexBuffer,
                         vk::MemoryPropertyFlagBits::eDeviceLocal,
                         vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd,
                         vertexBuffer,
                         vertexMemory);

    importExternalMemory(
        (void **)&points, cudaVertMem, vertexMemory, sim.getNumPoints() * sizeof(Point), vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd);

    sim.initSimulation(points);

    {
        void *data;
        vk::Buffer stagingBuffer;
        vk::DeviceMemory stagingMemory;
        vk::DeviceSize bufferSize = sim.getNumPoints() * sizeof(Point);

        createBuffer(bufferSize,
                     vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                     stagingBuffer,
                     stagingMemory);
        if (device.mapMemory(stagingMemory, 0, bufferSize, {}, &data) != vk::Result::eSuccess)
            throw std::runtime_error("Failed to map staging buffer memory");
        copyBuffer(stagingBuffer, vertexBuffer, bufferSize);

        for (size_t i = 0; i < sim.getNumPoints(); i++)
            indices.push_back(i);

        bufferSize = sizeof(indices[0]) * indices.size();
        createBuffer(bufferSize,
                     vk::BufferUsageFlagBits::eTransferSrc,
                     vk::MemoryPropertyFlagBits::eHostVisible | vk::MemoryPropertyFlagBits::eHostCoherent,
                     stagingBuffer,
                     stagingMemory);
        if (device.mapMemory(stagingMemory, 0, bufferSize, {}, &data) != vk::Result::eSuccess)
            throw std::runtime_error("Failed to map staging buffer memory");
        memcpy(data, indices.data(), ((size_t)bufferSize));

        createBuffer(bufferSize,
                     vk::BufferUsageFlagBits::eTransferDst | vk::BufferUsageFlagBits::eIndexBuffer,
                     vk::MemoryPropertyFlagBits::eDeviceLocal,
                     indexBuffer,
                     indexMemory);
        copyBuffer(stagingBuffer, indexBuffer, bufferSize);

        device.destroyBuffer(stagingBuffer);
        device.unmapMemory(stagingMemory);
        device.freeMemory(stagingMemory);
    }

    createExternalSemaphore(timelineSemaphore, vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd);
    importExternalSemaphore(cudaTimelineSemaphore, timelineSemaphore, vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd);
    sim.initPoints();
}

auto Renderer::updateSync() -> void {
    static uint64_t waitValue   = 0;
    static uint64_t signalValue = 1;

    if (device.waitSemaphores(vk::SemaphoreWaitInfo{.semaphoreCount = 1, .pSemaphores = &timelineSemaphore, .pValues = &waitValue},
                              std::numeric_limits<uint64_t>::max()) != vk::Result::eSuccess)
        throw std::runtime_error("Failed to wait semaphores");

    const auto imageResult = device.acquireNextImageKHR(swapChain, std::numeric_limits<uint64_t>::max(), presentationSemaphore, {});

    if (imageResult.result == vk::Result::eErrorOutOfDateKHR)
        recreateSwapChain();
    else if (imageResult.result != vk::Result::eSuccess && imageResult.result != vk::Result::eSuboptimalKHR)
        throw std::runtime_error("Failed to acquire swap chain images");

    updateUniformBuffer(imageResult.value);

    std::vector<vk::Semaphore> waitSemaphores{timelineSemaphore};
    std::vector<vk::PipelineStageFlags> waitStages{vk::PipelineStageFlagBits::eColorAttachmentOutput};

    std::vector<vk::Semaphore> signalSemaphores{timelineSemaphore};

    auto timelineInfo = vk::TimelineSemaphoreSubmitInfo{
        .waitSemaphoreValueCount = 1, .pWaitSemaphoreValues = &waitValue, .signalSemaphoreValueCount = 1, .pSignalSemaphoreValues = &signalValue};

    auto submitInfo = vk::SubmitInfo{.pNext                = &timelineInfo,
                                     .waitSemaphoreCount   = static_cast<uint32_t>(waitSemaphores.size()),
                                     .pWaitSemaphores      = waitSemaphores.data(),
                                     .pWaitDstStageMask    = waitStages.data(),
                                     .commandBufferCount   = 1,
                                     .pCommandBuffers      = &commandBuffers[imageResult.value],
                                     .signalSemaphoreCount = static_cast<uint32_t>(signalSemaphores.size()),
                                     .pSignalSemaphores    = signalSemaphores.data()};

    if (graphicsQueue.submit(1, &submitInfo, nullptr) != vk::Result::eSuccess)
        throw std::runtime_error("Failed to submit graphics queue");

    auto presentResult = presentQueue.presentKHR(vk::PresentInfoKHR{.waitSemaphoreCount = 1,
                                                                    .pWaitSemaphores    = &presentationSemaphore,
                                                                    .swapchainCount     = 1,
                                                                    .pSwapchains        = &swapChain,
                                                                    .pImageIndices      = &imageResult.value});
    if (presentResult == vk::Result::eErrorOutOfDateKHR || presentResult == vk::Result::eSuboptimalKHR || windowResize) {
        recreateSwapChain();
        windowResize = false;
    } else if (presentResult != vk::Result::eSuccess && presentResult != vk::Result::eSuboptimalKHR)
        throw std::runtime_error("Failed to acquire swapchain image");

    currentFrame++;
    waitValue   += 2;
    signalValue += 2;
}

auto Renderer::drawFrame() -> void {
    static float startTime = glfwGetTime();
    float currentTime      = glfwGetTime();
    float time             = currentTime - startTime;

    if (currentFrame == 0) {
        lastTime = startTime;
    }

    float frame_time = currentTime - lastTime;

    updateSync();

    static uint64_t waitValue   = 1;
    static uint64_t signalValue = 2;

    cudaExternalSemaphoreWaitParams waitParams = {.flags = 0};
    waitParams.params.fence.value              = waitValue;

    cudaExternalSemaphoreSignalParams signalParams = {.flags = 0};
    signalParams.params.fence.value                = signalValue;

    checkCudaErrors(cudaWaitExternalSemaphoresAsync(&cudaTimelineSemaphore, &waitParams, 1, stream));
    sim.stepSimulation(time, stream);
    checkCudaErrors(cudaSignalExternalSemaphoresAsync(&cudaTimelineSemaphore, &signalParams, 1, stream));

    waitValue   += 2;
    signalValue += 2;

    if (frame_time > 1) {
        // dont output fps first time bc inaccurate high makes written over value ugly
        const float fps = ((currentFrame - lastFrame) / frame_time) > 1000 ? 0 : ((currentFrame - lastFrame) / frame_time);

        std::cerr << "\rAverage FPS (over " << std::fixed << std::setprecision(2) << frame_time << " seconds): " << std::fixed << std::setprecision(2)
                  << fps << std::flush;
        lastFrame = currentFrame;
        lastTime  = currentTime;
    }
}

auto Renderer::mainLoop() -> void {
    while (!glfwWindowShouldClose(window)) {
        if (glfwGetKey(window, GLFW_KEY_ESCAPE) == GLFW_PRESS) // close window on escape
            glfwSetWindowShouldClose(window, true);

        glfwPollEvents();
        drawFrame();
    }
    device.waitIdle();
    std::cout << "\n"; // output new line at end so terminal doesnt end up on same line as fps
}

// ----------------------- UTIL FUNCTIONS ---------------------------------
auto Renderer::beginSingleTimeCommands() -> std::vector<vk::CommandBuffer> {
    auto commandBuffer = device.allocateCommandBuffers(
        vk::CommandBufferAllocateInfo{.commandPool = commandPool, .level = vk::CommandBufferLevel::ePrimary, .commandBufferCount = 1});
    commandBuffer[0].begin(vk::CommandBufferBeginInfo{.flags = vk::CommandBufferUsageFlagBits::eOneTimeSubmit});
    return commandBuffer;
}

auto Renderer::endSingleTimeCommands(std::vector<vk::CommandBuffer> commandBuffer) -> void {
    commandBuffer[0].end();
    const auto submitInfo = vk::SubmitInfo{.commandBufferCount = 1, .pCommandBuffers = &commandBuffer[0]};
    graphicsQueue.submit(submitInfo, {});
    graphicsQueue.waitIdle();
    device.freeCommandBuffers(commandPool, 1, commandBuffer.data());
}

auto Renderer::createImage(uint32_t width,
                           uint32_t height,
                           uint32_t mips,
                           vk::Format format,
                           vk::ImageTiling tiling,
                           vk::ImageUsageFlags usage,
                           vk::MemoryPropertyFlags properties,
                           vk::Image &image,
                           vk::DeviceMemory &imageMemory) -> void {
    const auto imageExtent = vk::Extent3D{static_cast<uint32_t>(width), static_cast<uint32_t>(height), 1};
    const auto imageinfo   = vk::ImageCreateInfo{.imageType     = vk::ImageType::e2D,
                                                 .format        = format,
                                                 .extent        = imageExtent,
                                                 .mipLevels     = mips,
                                                 .arrayLayers   = 1,
                                                 .samples       = vk::SampleCountFlagBits::e1,
                                                 .tiling        = tiling,
                                                 .usage         = usage,
                                                 .sharingMode   = vk::SharingMode::eExclusive,
                                                 .initialLayout = vk::ImageLayout::eUndefined};

    if (device.createImage(&imageinfo, nullptr, &image) != vk::Result::eSuccess)
        throw std::runtime_error("Failed to create texture image");

    const auto memoryRequirements = device.getImageMemoryRequirements(image);
    const auto allocInfo          = vk::MemoryAllocateInfo{.allocationSize  = memoryRequirements.size,
                                                           .memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, properties)};
    if (device.allocateMemory(&allocInfo, nullptr, &imageMemory) != vk::Result::eSuccess)
        throw std::runtime_error("Failed to allocate texture memory");

    device.bindImageMemory(image, imageMemory, 0);
}

auto Renderer::createBuffer(
    vk::DeviceSize size, vk::BufferUsageFlags usage, vk::MemoryPropertyFlags properties, vk::Buffer &buffer, vk::DeviceMemory &bufferMemory) -> void {
    buffer = device.createBuffer(vk::BufferCreateInfo{.size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive});

    const auto memoryRequirements = device.getBufferMemoryRequirements(buffer);
    const auto memoryAllocInfo    = vk::MemoryAllocateInfo{.allocationSize  = memoryRequirements.size,
                                                           .memoryTypeIndex = findMemoryType(memoryRequirements.memoryTypeBits, properties)};
    if (device.allocateMemory(&memoryAllocInfo, nullptr, &bufferMemory) != vk::Result::eSuccess)
        throw std::runtime_error("Failed to allocate  buffer memory");

    device.bindBufferMemory(buffer, bufferMemory, 0);
}

auto Renderer::copyBuffer(vk::Buffer srcBuffer, vk::Buffer dstBuffer, vk::DeviceSize size) -> void {
    auto commandBuffer = beginSingleTimeCommands();

    const vk::BufferCopy copyRegion{.srcOffset = 0, .dstOffset = 0, .size = size};
    commandBuffer[0].copyBuffer(srcBuffer, dstBuffer, 1, &copyRegion);

    endSingleTimeCommands(commandBuffer);
}

auto Renderer::importExternalSemaphore(cudaExternalSemaphore_t &cudaSem, vk::Semaphore &vkSem, vk::ExternalSemaphoreHandleTypeFlagBits handleType)
    -> void {

    auto semaphoreGetFdInfoKHR = vk::SemaphoreGetFdInfoKHR{.semaphore = vkSem, .handleType = handleType};

    auto fpGetSemaphoreFdKHR = (PFN_vkGetSemaphoreFdKHR)vkGetDeviceProcAddr(device, "vkGetSemaphoreFdKHR");
    if (!fpGetSemaphoreFdKHR) {
        throw std::runtime_error("Failed to retrieve vkGetMemoryWin32HandleKHR!");
    }
    int fd = 0;
    if (fpGetSemaphoreFdKHR(device, reinterpret_cast<VkSemaphoreGetFdInfoKHR *>(&semaphoreGetFdInfoKHR), &fd) != VK_SUCCESS) {
        throw std::runtime_error("Failed to retrieve handle for buffer!");
    }

    cudaExternalSemaphoreHandleDesc externalSemaphoreHandleDesc = {.flags = 0};
    if (handleType & vk::ExternalSemaphoreHandleTypeFlagBits::eOpaqueFd)
        externalSemaphoreHandleDesc.type = cudaExternalSemaphoreHandleTypeTimelineSemaphoreFd;
    else
        throw std::runtime_error("Unknown external semaphore handle type");

    externalSemaphoreHandleDesc.handle.fd = fd;
    checkCudaErrors(cudaImportExternalSemaphore(&cudaSem, &externalSemaphoreHandleDesc));
}

auto Renderer::createExternalSemaphore(vk::Semaphore &semaphore, vk::ExternalSemaphoreHandleTypeFlagBits handleType) -> void {

    const auto timelineCreateInfo        = vk::SemaphoreTypeCreateInfo{.semaphoreType = vk::SemaphoreType::eTimeline, .initialValue = 0};
    const auto exportSemaphoreCreateInfo = vk::ExportSemaphoreCreateInfo{.pNext = &timelineCreateInfo, .handleTypes = handleType};
    const auto semaphoreInfo             = vk::SemaphoreCreateInfo{.pNext = &exportSemaphoreCreateInfo};

    if (device.createSemaphore(&semaphoreInfo, nullptr, &semaphore) != vk::Result::eSuccess)
        throw std::runtime_error("Failed to create Cuda-Vulkan interop semaphore");
}

auto Renderer::createExternalBuffer(vk::DeviceSize size,
                                    vk::BufferUsageFlags usage,
                                    vk::MemoryPropertyFlags properties,
                                    vk::ExternalMemoryHandleTypeFlagsKHR extMemHandleType,
                                    vk::Buffer &buffer,
                                    vk::DeviceMemory &bufferMemory) -> void {
    const auto externalMemoryBufferInfo = vk::ExternalMemoryBufferCreateInfo{.handleTypes = extMemHandleType};
    const auto bufferInfo =
        vk::BufferCreateInfo{.pNext = &externalMemoryBufferInfo, .size = size, .usage = usage, .sharingMode = vk::SharingMode::eExclusive};

    if (device.createBuffer(&bufferInfo, nullptr, &buffer) != vk::Result::eSuccess)
        throw std::runtime_error("Failed to Create Buffer");

    const auto memRequirements          = device.getBufferMemoryRequirements(buffer);
    const auto exportMemoryAllocateInfo = vk::ExportMemoryAllocateInfoKHR{.handleTypes = vk::ExternalMemoryHandleTypeFlagBits::eOpaqueFd};
    const auto allocInfo                = vk::MemoryAllocateInfo{.pNext           = &exportMemoryAllocateInfo,
                                                                 .allocationSize  = memRequirements.size,
                                                                 .memoryTypeIndex = findMemoryType(memRequirements.memoryTypeBits, properties)};
    if (device.allocateMemory(&allocInfo, nullptr, &bufferMemory) != vk::Result::eSuccess)
        throw std::runtime_error("Failed to allocate external buffer memory");
    device.bindBufferMemory(buffer, bufferMemory, 0);
}

auto Renderer::importExternalMemory(void **cudaPtr,
                                    cudaExternalMemory_t &cudaMem,
                                    vk::DeviceMemory &vkMem,
                                    vk::DeviceSize size,
                                    vk::ExternalMemoryHandleTypeFlagBits handleType) -> void {
    cudaExternalMemoryHandleDesc externalMemoryHandleDesc = {.type = cudaExternalMemoryHandleTypeOpaqueFd, .size = size};
    externalMemoryHandleDesc.handle.fd                    = (int)(uintptr_t)getMemHandle(vkMem, handleType);

    checkCudaErrors(cudaImportExternalMemory(&cudaMem, &externalMemoryHandleDesc));

    cudaExternalMemoryBufferDesc externalMemBufferDesc = {};
    externalMemBufferDesc.offset                       = 0;
    externalMemBufferDesc.size                         = size;
    externalMemBufferDesc.flags                        = 0;

    checkCudaErrors(cudaExternalMemoryGetMappedBuffer(cudaPtr, cudaMem, &externalMemBufferDesc));
}

auto Renderer::getMemHandle(vk::DeviceMemory memory, vk::ExternalMemoryHandleTypeFlagBits handleType) -> void * {
    int fd = -1;

    auto vkMemoryGetFdInfoKHR = vk::MemoryGetFdInfoKHR{.memory = memory, .handleType = handleType};
    auto fpGetMemoryFdKHR     = (PFN_vkGetMemoryFdKHR)vkGetInstanceProcAddr(instance, "vkGetMemoryFdKHR");

    if (!fpGetMemoryFdKHR)
        throw std::runtime_error("Failed to retrieve vkGetMemoryWin32HandleKHR!");

    if (fpGetMemoryFdKHR(device, reinterpret_cast<VkMemoryGetFdInfoKHR *>(&vkMemoryGetFdInfoKHR), &fd) != VK_SUCCESS)
        throw std::runtime_error("Failed to retrieve handle for buffer!");
    else
        return (void *)(uintptr_t)fd;
}

auto Renderer::findSupportedFormat(const std::vector<vk::Format> &candidates, vk::ImageTiling tiling, vk::FormatFeatureFlags features) -> vk::Format {
    for (const vk::Format &format : candidates) {
        const auto properties = physicalDevice.getFormatProperties(format);

        if (tiling == vk::ImageTiling::eLinear && (properties.linearTilingFeatures & features) == features)
            return format;
        else if (tiling == vk::ImageTiling::eOptimal && (properties.optimalTilingFeatures & features) == features)
            return format;
    }

    throw std::runtime_error("Failed to find supported format");
}

constexpr auto Renderer::findMemoryType(uint32_t typeFilter, vk::MemoryPropertyFlags properties) -> uint32_t {
    const auto memProperties = physicalDevice.getMemoryProperties();

    for (uint32_t i = 0; i < memProperties.memoryTypeCount; i++) {
        if (typeFilter & (1 << i) && (memProperties.memoryTypes[i].propertyFlags & properties) == properties)
            return i;
    }

    throw std::runtime_error("Failed to find suitable memory");
}
