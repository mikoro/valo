// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include <cstdio>
#include <cuda_runtime.h>

#include "Cuda/CudaKernels.h"
#include "Filters/Filter.h"

using namespace Raycer;

__global__ void testKernel(Filter* filter)
{
	printf("Test!\n");
	filter[1].getWeight(23.0f);
}

void cudaTest(Filter* filter)
{
	testKernel<<<1,1>>>(filter);
	cudaDeviceSynchronize();
}
