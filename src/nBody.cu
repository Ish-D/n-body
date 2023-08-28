#include "nBody.h"

__global__ void step(nBody::Point *points, unsigned int numPoints, float time) {
  const float freq = 4.0f;
  const size_t stride = gridDim.x * blockDim.x;

  // Iterate through the entire array in a way that is independent of the grid configuration
  for (size_t tid = blockIdx.x * blockDim.x + threadIdx.x; tid < numPoints; tid+= stride) {
    const size_t y = tid / numPoints;
    const size_t x = tid - y * numPoints;

    const float u = ((2.0f * x) / numPoints) - 1.0f;
    const float v = ((2.0f * y) / numPoints) - 1.0f;

    points[tid].pos.x = sinf(u * freq + time);
    points[tid].pos.y = 1.5*tid/numPoints;
    // points[tid].pos.y = cosf(v * freq + time);
  }
}

nBody::nBody(size_t _numPoints, Point* _points) : numPoints(_numPoints), points(_points), pointsTemp(_points) {}
nBody::~nBody() { points = NULL; }

void nBody::initCudaLaunchConfig(int device) {
  cudaDeviceProp prop = {};
  checkCudaErrors(cudaSetDevice(device));
  checkCudaErrors(cudaGetDeviceProperties(&prop, device));

  m_threads = prop.warpSize;

  // Use  occupancy calculator and fill the gpu as best as we can
  checkCudaErrors(cudaOccupancyMaxActiveBlocksPerMultiprocessor(&m_blocks, step, prop.warpSize, 0));
  m_blocks *= prop.multiProcessorCount;
  
  // Clamp the blocks to the minimum needed for this height/width
  m_blocks = std::min(m_blocks, (int)((numPoints + m_threads - 1) / m_threads));
}

int nBody::initCuda(uint8_t *vkDeviceUUID, size_t UUID_SIZE) {
  int current_device = 0;
  int device_count = 0;
  int devices_prohibited = 0;

  cudaDeviceProp deviceProp;
  checkCudaErrors(cudaGetDeviceCount(&device_count));

  if (device_count == 0) {
    fprintf(stderr, "CUDA error: no devices supporting CUDA.\n");
    exit(EXIT_FAILURE);
  }

  // Find the GPU which is selected by Vulkan
  while (current_device < device_count) {
    cudaGetDeviceProperties(&deviceProp, current_device);

    if ((deviceProp.computeMode != cudaComputeModeProhibited)) {
      // Compare the cuda device UUID with vulkan UUID
      int ret = memcmp((void *)&deviceProp.uuid, vkDeviceUUID, UUID_SIZE);
      if (ret == 0) {
        checkCudaErrors(cudaSetDevice(current_device));
        checkCudaErrors(cudaGetDeviceProperties(&deviceProp, current_device));
        printf("GPU Device %d: \"%s\" with compute capability %d.%d\n\n", current_device, deviceProp.name, deviceProp.major, deviceProp.minor);

        return current_device;
      }
    }

    else {
      devices_prohibited++;
    }

    current_device++;
  }

  if (devices_prohibited == device_count) {
    fprintf(stderr, "CUDA error:" " No Vulkan-CUDA Interop capable GPU found.\n");
    exit(EXIT_FAILURE);
  }

  return -1;
}

void nBody::initSimulation(Point* _points) {
  points = _points;
}

void nBody::initPoints() {
  checkCudaErrors(cudaMemcpy(points, pointsTemp, numPoints*sizeof(Point), cudaMemcpyHostToDevice));
}

void nBody::stepSimulation(float time, cudaStream_t stream) {
  step<<<m_blocks, m_threads, 0, stream>>>(points, numPoints, time);
  getLastCudaError("Failed to launch CUDA simulation");
}