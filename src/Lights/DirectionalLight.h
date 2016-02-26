// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

#include "Lights/Light.h"
#include "Rendering/Color.h"
#include "Math/Vector3.h"

namespace Raycer
{
	class Scene;
	class Intersection;
	class Random;
	class Vector3;

	class DirectionalLight : public Light
	{
	public:
		
		void initialize() override;
		bool hasDirection() const override;

		Color getColor(const Scene& scene, const Intersection& intersection, Random& random) const override;
		Vector3 getDirection(const Intersection& intersection) const override;
		
		Color color;
		Vector3 direction;

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(color),
				CEREAL_NVP(direction));
		}
	};
}
