// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Samplers/RegularSampler.h"
#include "Math/Vector2.h"

using namespace Raycer;

double RegularSampler::getSample(uint64_t x, uint64_t n, uint64_t permutation, Random& random)
{
	(void)permutation;
	(void)random;

	assert(x < n);

	return (double(x) + 0.5) / double(n);
}

Vector2 RegularSampler::getSquareSample(uint64_t x, uint64_t y, uint64_t nx, uint64_t ny, uint64_t permutation, Random& random)
{
	(void)permutation;
	(void)random;

	assert(x < nx && y < ny);

	Vector2 result;

	result.x = (double(x) + 0.5) / double(nx);
	result.y = (double(y) + 0.5) / double(ny);

	return result;
}
