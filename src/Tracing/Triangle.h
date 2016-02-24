// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include "cereal/cereal.hpp"

#include "Tracing/AABB.h"
#include "Math/Vector3.h"
#include "Math/Vector2.h"

namespace Raycer
{
	class Ray;
	class Intersection;
	class Material;
	class Random;

	class Triangle
	{
	public:

		void initialize();
		bool intersect(const Ray& ray, Intersection& intersection) const;
		Intersection getRandomIntersection(Random& random) const;
		AABB getAABB() const;
		float getArea() const;

		uint64_t id = 0;
		uint64_t materialId = 0;

		Vector3 vertices[3];
		Vector3 normals[3];
		Vector2 texcoords[3];
		Vector3 normal;
		Vector3 tangent;
		Vector3 bitangent;
		
		Material* material = nullptr;

	private:

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(id),
				CEREAL_NVP(materialId),
				CEREAL_NVP(vertices),
				CEREAL_NVP(normals),
				CEREAL_NVP(texcoords));
		}
	};
}
