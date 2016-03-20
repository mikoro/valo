// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

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

		Color calculateRadiance(const Scene& scene, const Ray& viewRay, Random& random);

		uint32_t getSampleCount() const;
		std::string getName() const;

		static Color calculateDirectLight(const Scene& scene, const Intersection& intersection, Random& random);
		static void calculateNormalMapping(Intersection& intersection);

		IntegratorType type = IntegratorType::DOT;

		DotIntegrator dotIntegrator;
		PathIntegrator pathIntegrator;

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(type),
				CEREAL_NVP(pathIntegrator));
		}
	};
}
