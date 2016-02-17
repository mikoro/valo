// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Samplers/CenterSampler.h"
#include "Math/Vector2.h"

using namespace Raycer;

float CenterSampler::getSample(uint64_t x, uint64_t n, uint64_t permutation, Random& random)
{
	(void)x;
	(void)n;
	(void)permutation;
	(void)random;

	assert(x < n);

	return 0.5f;
}

Vector2 CenterSampler::getSquareSample(uint64_t x, uint64_t y, uint64_t nx, uint64_t ny, uint64_t permutation, Random& random)
{
	(void)x;
	(void)y;
	(void)nx;
	(void)ny;
	(void)permutation;
	(void)random;

	assert(x < nx && y < ny);

	return Vector2(0.5f, 0.5f);
}
