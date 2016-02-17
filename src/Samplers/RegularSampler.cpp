// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Samplers/RegularSampler.h"
#include "Math/Vector2.h"

using namespace Raycer;

float RegularSampler::getSample(uint64_t x, uint64_t n, uint64_t permutation, Random& random)
{
	(void)permutation;
	(void)random;

	assert(x < n);

	return (float(x) + 0.5f) / float(n);
}

Vector2 RegularSampler::getSquareSample(uint64_t x, uint64_t y, uint64_t nx, uint64_t ny, uint64_t permutation, Random& random)
{
	(void)permutation;
	(void)random;

	assert(x < nx && y < ny);

	Vector2 result;

	result.x = (float(x) + 0.5f) / float(nx);
	result.y = (float(y) + 0.5f) / float(ny);

	return result;
}
