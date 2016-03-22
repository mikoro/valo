// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#ifdef USE_CUDA
#include <cuda_runtime.h>
#endif

#include "Utils/Allocator.h"

using namespace Raycer;

void* Allocator::operator new(size_t len)
{
	return Allocator::malloc(len);
}

void Allocator::operator delete(void* ptr)
{
	Allocator::free(ptr);
}

void* Allocator::malloc(size_t len)
{
	void* ptr;

#ifdef USE_CUDA
	cudaMallocManaged(&ptr, len);
	cudaDeviceSynchronize();
#else
	ptr = ::malloc(len);
#endif

	return ptr;
}

void Allocator::free(void* ptr)
{
	if (ptr == nullptr)
		return;

#ifdef USE_CUDA
	cudaDeviceSynchronize();
	cudaFree(ptr);
#else
	::free(ptr);
#endif
}
