// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Core/Common.h"
#include "Integrators/DotIntegrator.h"
#include "Integrators/PathIntegrator.h"

namespace Raycer
{
	class Color;
	class Scene;
	class Intersection;
	class Ray;
	class Random;

	enum class IntegratorType { DOT, PATH };

	class Integrator
	{
	public:

		CUDA_CALLABLE Color calculateRadiance(const Scene& scene, const Ray& viewRay, Random& random) const;

		uint32_t getSampleCount() const;
		std::string getName() const;

		CUDA_CALLABLE static Color calculateDirectLight(const Scene& scene, const Intersection& intersection, const Vector3& in, Random& random);

		IntegratorType type = IntegratorType::DOT;

		DotIntegrator dotIntegrator;
		PathIntegrator pathIntegrator;
	};
}
