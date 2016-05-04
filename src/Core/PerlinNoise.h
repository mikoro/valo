// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Core/CudaAlloc.h"
#include "Core/Common.h"

/*

http://mrl.nyu.edu/~perlin/noise/
http://mrl.nyu.edu/~perlin/paper445.pdf

getNoise returns values between 0.0 - 1.0
getFbmNoise return values between 0.0 - inf

*/

namespace Raycer
{
	class PerlinNoise
	{
	public:

		PerlinNoise();

		void initialize(uint32_t seed);
		CUDA_CALLABLE float getNoise(float x, float y, float z) const;
		CUDA_CALLABLE float getFbmNoise(uint32_t octaves, float lacunarity, float persistence, float x, float y, float z) const;

	private:

		CUDA_CALLABLE float fade(float t) const;
		CUDA_CALLABLE float lerp(float t, float a, float b) const;
		CUDA_CALLABLE float grad(uint32_t hash, float x, float y, float z) const;

		CudaAlloc<uint32_t> permutationsAlloc;
	};
}
