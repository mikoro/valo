// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Samplers/JitteredSampler.h"
#include "Math/Vector2.h"

using namespace Raycer;

float JitteredSampler::getSample(uint64_t x, uint64_t n, uint64_t permutation, Random& random)
{
	(void)permutation;

	assert(x < n);

	return (float(x) + random.getFloat()) / float(n);
}

Vector2 JitteredSampler::getSquareSample(uint64_t x, uint64_t y, uint64_t nx, uint64_t ny, uint64_t permutation, Random& random)
{
	(void)permutation;

	assert(x < nx && y < ny);

	Vector2 result;

	result.x = (float(x) + random.getFloat()) / float(nx);
	result.y = (float(y) + random.getFloat()) / float(ny);

	return result;
}
