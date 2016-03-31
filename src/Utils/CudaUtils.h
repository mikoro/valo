// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#ifdef USE_CUDA
#include <cuda_runtime.h>
#else
#define cudaError_t int
#endif

namespace Raycer
{
	class CudaUtils
	{
	public:

		static void checkError(cudaError_t code);
	};
}
