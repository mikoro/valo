// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Samplers/PoissonDiscSampler.h"
#include "Math/Vector2.h"
#include "Utils/PoissonDisc.h"

using namespace Raycer;

float PoissonDiscSampler::getSample(uint64_t x, uint64_t n, uint64_t permutation, Random& random)
{
	(void)x;
	(void)n;
	(void)permutation;
	(void)random;

	assert(x < n);

	return 0.0f;
}

Vector2 PoissonDiscSampler::getSquareSample(uint64_t x, uint64_t y, uint64_t nx, uint64_t ny, uint64_t permutation, Random& random)
{
	(void)x;
	(void)y;
	(void)nx;
	(void)ny;
	(void)permutation;
	(void)random;

	assert(x < nx && y < ny);

	return Vector2();
}

void PoissonDiscSampler::generateSamples2D(uint64_t sampleCountSqrt, Random& random)
{
	(void)random;

	PoissonDisc poissonDisc;
	samples2D = poissonDisc.generate(sampleCountSqrt, sampleCountSqrt, 1.0f / float(M_SQRT2), 30, true); // minDistance is just a guess to get about sampleCountSqrt^2 samples
}
