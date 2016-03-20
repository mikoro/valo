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

	class DotIntegrator
	{
	public:

		Color calculateRadiance(const Scene& scene, const Ray& viewRay, Random& random);

		uint32_t getSampleCount() const;

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
		}
	};
}
