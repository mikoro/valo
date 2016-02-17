// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Samplers/Sampler.h"

namespace Raycer
{
	class CenterSampler : public Sampler
	{
	public:

		float getSample(uint64_t x, uint64_t n, uint64_t permutation, Random& random) override;
		Vector2 getSquareSample(uint64_t x, uint64_t y, uint64_t nx, uint64_t ny, uint64_t permutation, Random& random) override;
	};
}
