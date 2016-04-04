// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "Core/Common.h"
#include "Integrators/PathIntegrator.h"
#include "Integrators/DotIntegrator.h"
#include "Integrators/AmbientOcclusionIntegrator.h"
#include "Integrators/DirectLightIntegrator.h"

namespace Raycer
{
	class Color;
	class Scene;
	class Intersection;
	class Ray;
	class Random;

	enum class IntegratorType { PATH, DOT, AMBIENT_OCCLUSION, DIRECT_LIGHT };

	class Integrator
	{
	public:

		CUDA_CALLABLE Color calculateLight(const Scene& scene, const Intersection& intersection, const Ray& ray, Random& random) const;

		std::string getName() const;

		CUDA_CALLABLE static Color calculateDirectLight(const Scene& scene, const Intersection& intersection, const Vector3& in, Random& random);

		IntegratorType type = IntegratorType::PATH;

		PathIntegrator pathIntegrator;
		DotIntegrator dotIntegrator;
		AmbientOcclusionIntegrator aoIntegrator;
		DirectLightIntegrator directIntegrator;
	};
}
