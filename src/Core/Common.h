// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <cuda_runtime.h>

#define RAYCER_VERSION "2.0.0"
#define CACHE_LINE_SIZE 64

#ifdef _MSC_VER
#define ALIGN(x) __declspec(align(x))
#elif __GNUC__
#define ALIGN(x) __attribute__ ((aligned(x)))
#endif

#define MIN(a,b) (((a)<(b))?(a):(b))
#define MAX(a,b) (((a)>(b))?(a):(b))

#define RAYCER_MALLOC(x) malloc(x)
#define RAYCER_FREE(x) if (x != nullptr) { free(x); x = nullptr; }

#define CUDA_CALLABLE __host__ __device__
