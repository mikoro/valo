// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

namespace Raycer
{
	class Vector3;
	class Intersection;
	class Random;
	class Color;
	class Material;

	class DiffuseMaterial
	{
	public:

		Vector3 getDirection(const Material& material, const Intersection& intersection, Random& random);
		Color getBrdf(const Material& material, const Intersection& intersection, const Vector3& out);
		float getPdf(const Material& material, const Intersection& intersection, const Vector3& out);

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
		}
	};
}
