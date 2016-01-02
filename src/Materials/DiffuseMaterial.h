// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

#include "Materials/Material.h"
#include "Rendering/Color.h"

namespace Raycer
{
	class Scene;
	class Intersection;
	class Light;
	class Random;
	class RandomSampler;
	class Vector3;

	class DiffuseMaterial : public Material
	{
	public:

		Color getColor(const Scene& scene, const Intersection& intersection, const Light& light, Random& random) override;

		Vector3 getSampleDirection(const Intersection& intersection, RandomSampler& sampler, Random& random) override;
		double getDirectionProbability(const Intersection& intersection, const Vector3& out) override;
		Color getBrdf(const Intersection& intersection, const Vector3& out) override;

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(cereal::make_nvp("material", cereal::base_class<Material>(this)));
		}
	};
}
