// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Samplers/JitteredSampler.h"
#include "Math/Vector2.h"

using namespace Raycer;

double JitteredSampler::getSample1D(uint64_t x, uint64_t n, uint64_t permutation, Random& random)
{
	(void)permutation;

	assert(x < n);

	return (double(x) + random.getDouble()) / double(n);
}

Vector2 JitteredSampler::getSample2D(uint64_t x, uint64_t y, uint64_t nx, uint64_t ny, uint64_t permutation, Random& random)
{
	(void)permutation;

	assert(x < nx && y < ny);

	Vector2 result;

	result.x = (double(x) + random.getDouble()) / double(nx);
	result.y = (double(y) + random.getDouble()) / double(ny);

	return result;
}
