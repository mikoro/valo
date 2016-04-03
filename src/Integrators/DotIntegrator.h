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

	class DotIntegrator
	{
	public:

		CUDA_CALLABLE Color calculateRadiance(const Scene& scene, const Ray& viewRay, Random& random) const;

		uint32_t getSampleCount() const;
	};
}
