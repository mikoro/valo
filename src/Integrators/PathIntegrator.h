// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

namespace Raycer
{
	class Color;
	class Scene;
	class Ray;
	class Random;

	class PathIntegrator
	{
	public:

		Color calculateRadiance(const Scene& scene, const Ray& viewRay, Random& random);

		uint32_t getSampleCount() const;

		uint32_t pathSamples = 1;
		uint32_t minPathLength = 3;
		float terminationProbability = 0.5f;

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(pathSamples),
				CEREAL_NVP(minPathLength),
				CEREAL_NVP(terminationProbability));
		}
	};
}
