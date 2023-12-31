#pragma once

#include <cuda_runtime_api.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>

class nBody {

    size_t numPoints;
    int m_blocks, m_threads;

  public:
    struct vec3 {
        float x = 0.0f;
        float y = 0.0f;
        float z = 0.0f;
    };

    struct Color {
        float r = 0.25f;
        float g = 0.0f;
        float b = 0.8f;
    };

    struct Point {
        vec3 pos{};
        float size = 20.0f;
        Color color{};

        vec3 velocity{};
        vec3 acceleration{};
    };

    Point *points;
    Point *pointsTemp;

    nBody(size_t _numPoints, Point *_points);
    ~nBody();
    void initSimulation(Point *_points);
    void initPoints();
    void stepSimulation(float time, cudaStream_t stream = 0);
    void initCudaLaunchConfig(int device);
    int initCuda(uint8_t *vkDeviceUUID, size_t UUID_SIZE);

    size_t getNumPoints() const { return numPoints; }
    Point *getPoints() const { return points; }
};

template <typename T> void check(T result, char const *const func, const char *const file, int const line) {
    if (result) {
        fprintf(stderr, "CUDA error at %s:%d code=%d(%s) \"%s\" \n", file, line, static_cast<unsigned int>(result), cudaGetErrorName(result), func);
        exit(EXIT_FAILURE);
    }
}

#define checkCudaErrors(val) check((val), #val, __FILE__, __LINE__)

// This will output the proper error string when calling cudaGetLastError
#define getLastCudaError(msg) __getLastCudaError(msg, __FILE__, __LINE__)

inline void __getLastCudaError(const char *errorMessage, const char *file, const int line) {
    cudaError_t err = cudaGetLastError();

    if (cudaSuccess != err) {
        fprintf(stderr,
                "%s(%i) : getLastCudaError() CUDA error :"
                " %s : (%d) %s.\n",
                file,
                line,
                errorMessage,
                static_cast<int>(err),
                cudaGetErrorString(err));
        exit(EXIT_FAILURE);
    }
}