// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Core/Common.h"

namespace Raycer
{
	class Color;
	class Vector2;
	class Vector3;

	// http://www.pcg-random.org
	class RandomGeneratorPCG
	{
	public:

		CUDA_CALLABLE RandomGeneratorPCG();
		CUDA_CALLABLE explicit RandomGeneratorPCG(uint64_t seed);

		CUDA_CALLABLE void seed(uint64_t seed);

		typedef uint32_t result_type;
		CUDA_CALLABLE static result_type min() { return 0; };
		CUDA_CALLABLE static result_type max() { return INT_MAX; };
		CUDA_CALLABLE result_type operator()();

	private:

		uint64_t state;
		uint64_t inc;
	};

	class Random
	{
	public:

		CUDA_CALLABLE Random();
		CUDA_CALLABLE explicit Random(uint64_t seed);

		CUDA_CALLABLE void seed(uint64_t seed);

		CUDA_CALLABLE uint32_t getUint32();
		CUDA_CALLABLE uint32_t getUint32(uint32_t max);
		CUDA_CALLABLE uint32_t getUint32(uint32_t min, uint32_t max);
		CUDA_CALLABLE float getFloat();

		CUDA_CALLABLE Color getColor(bool randomAlpha = false);
		CUDA_CALLABLE Vector2 getVector2();
		CUDA_CALLABLE Vector3 getVector3();

	private:

		RandomGeneratorPCG generator;
	};
}
