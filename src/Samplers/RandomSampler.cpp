// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Samplers/RandomSampler.h"
#include "Math/Vector2.h"

using namespace Raycer;

double RandomSampler::getSample(uint64_t x, uint64_t n, uint64_t permutation, Random& random)
{
	(void)x;
	(void)n;
	(void)permutation;

	return random.getDouble();
}

Vector2 RandomSampler::getSquareSample(uint64_t x, uint64_t y, uint64_t nx, uint64_t ny, uint64_t permutation, Random& random)
{
	(void)x;
	(void)y;
	(void)nx;
	(void)ny;
	(void)permutation;

	return Vector2(random.getDouble(), random.getDouble());
}
