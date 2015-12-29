// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <memory>

#include "cereal/cereal.hpp"

#include "Lights/Light.h"
#include "Samplers/Sampler.h"
#include "Rendering/Color.h"

namespace Raycer
{
	class Scene;
	class Intersection;
	class Random;
	class Vector3;

	class AmbientLight : public Light
	{
	public:
		
		void initialize() override;
		bool hasDirection() const override;

		Color getColor(const Scene& scene, const Intersection& intersection, Random& random) const override;
		Vector3 getDirection(const Intersection& intersection) const override;
		
		Color color;
		bool occlusion = false;
		SamplerType samplerType = SamplerType::CMJ;
		uint64_t sampleCountSqrt = 3;
		double maxSampleDistance = 1.0;
		double sampleDistribution = 1.0;

	private:

		std::shared_ptr<Sampler> sampler = nullptr;

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(color),
				CEREAL_NVP(occlusion),
				CEREAL_NVP(samplerType),
				CEREAL_NVP(sampleCountSqrt),
				CEREAL_NVP(maxSampleDistance),
				CEREAL_NVP(sampleDistribution));
		}
	};
}
