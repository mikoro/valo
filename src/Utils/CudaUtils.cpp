// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "Utils/CudaUtils.h"

using namespace Raycer;

void CudaUtils::checkError(cudaError_t code)
{
	(void)code;

#ifdef USE_CUDA
	if (code != cudaSuccess)
		throw std::runtime_error(tfm::format("Cuda error: %s", cudaGetErrorString(code)));
#endif
}
