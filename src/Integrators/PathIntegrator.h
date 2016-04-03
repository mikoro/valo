// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Core/Common.h"

namespace Raycer
{
	class Color;
	class Scene;
	class Intersection;
	class Ray;
	class Random;

	class PathIntegrator
	{
	public:

		CUDA_CALLABLE Color calculateLight(const Scene& scene, const Intersection& intersection, const Ray& ray, Random& random) const;

		uint32_t getSampleCount() const;

		uint32_t pathSamples = 1;
		uint32_t minPathLength = 3;
		uint32_t maxPathLength = 10;
		float terminationProbability = 0.2f;
	};
}
