#pragma once

#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime_api.h>

class nBody {
    public:
        nBody();
        int blocks, threads;
};