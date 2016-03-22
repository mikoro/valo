// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Core/Common.h"

namespace Raycer
{
	class Color;
	class Scene;
	class Ray;
	class Random;

	class PathIntegrator
	{
	public:

		CUDA_CALLABLE Color calculateRadiance(const Scene& scene, const Ray& viewRay, Random& random);

		uint32_t getSampleCount() const;

		uint32_t pathSamples = 1;
		uint32_t minPathLength = 3;
		uint32_t maxPathLength = 10;
		float terminationProbability = 0.5f;
	};
}
