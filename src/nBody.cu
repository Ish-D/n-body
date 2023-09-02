#include "nBody.h"

__constant__ float damping = 1.0f;
__constant__ float softening = 0.05f;

__device__ nBody::vec3 bodyBodyInteraction(nBody::Point point, nBody::Point pbj, nBody::vec3 ai) {
    nBody::vec3 bi = point.pos;
    nBody::vec3 bj = pbj.pos;

    nBody::vec3 r{};
    r.x                = bj.x - bi.x;
    r.y                = bj.y - bi.y;
    r.z                = bj.z - bi.z;             

    float distSqr      = r.x * r.x + r.y * r.y + r.z * r.z + softening;
    float distSixth    = distSqr * distSqr * distSqr;
    float invDistCube  = 1.0f / sqrtf(distSixth);
    float s            = pbj.size * invDistCube; 
    
    ai.x              += r.x * s;
    ai.y              += r.y * s;
    ai.z              += r.z * s;

    return ai;
}

__device__ nBody::vec3 gravitation(nBody::Point point, nBody::Point* points) {
    nBody::vec3 acceleration{};

    for (int i = 0; i < blockDim.x; i++) 
        acceleration = bodyBodyInteraction(point, points[i], acceleration);

    return acceleration;
}

__global__ void step(nBody::Point *points, unsigned int numPoints, float time) {
  int index  = __mul24(blockIdx.x, blockDim.x) + threadIdx.x;
  if (index >= numPoints) return;

  nBody::vec3 pos = points[index].pos;
  nBody::vec3 velocity = points[index].velocity;

  nBody::vec3 force = gravitation(points[index], points);

  velocity.x += force.x * time;
  velocity.y += force.y * time;
  velocity.z += force.z * time;  

  velocity.x *= damping;
  velocity.y *= damping;
  velocity.z *= damping;

  pos.x += velocity.x * time;
  pos.y += velocity.y * time;
  pos.z += velocity.z * time;

  points[index].pos = pos;
  points[index].velocity = velocity;
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

    else devices_prohibited++;

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