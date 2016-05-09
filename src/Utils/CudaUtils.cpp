// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "tinyformat/tinyformat.h"

#include "Utils/CudaUtils.h"

using namespace Valo;

void CudaUtils::checkError(cudaError_t code, const std::string& message)
{
	(void)code;
	(void)message;

#ifdef USE_CUDA
	if (code != cudaSuccess)
		throw std::runtime_error(tfm::format("Cuda error: %s: %s", message, cudaGetErrorString(code)));
#endif
}
