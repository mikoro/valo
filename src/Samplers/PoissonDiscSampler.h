﻿// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <random>

#include "Samplers/Sampler.h"

namespace Raycer
{
	class PoissonDiscSampler : public Sampler
	{
	public:

		double getSample1D(uint64_t x, uint64_t n, uint64_t permutation, std::mt19937& generator) override;
		Vector2 getSample2D(uint64_t x, uint64_t y, uint64_t nx, uint64_t ny, uint64_t permutation, std::mt19937& generator) override;

		void generateSamples2D(uint64_t sampleCountSqrt, std::mt19937& generator) override;
	};
}