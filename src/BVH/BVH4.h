// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <vector>

#include "cereal/cereal.hpp"

#include "BVH/BVHCommon.h"
#include "Core/Triangle.h"

namespace Raycer
{
	class Triangle;
	class Scene;
	class Ray;
	class Intersection;

	class BVH4
	{
	public:

		void build(std::vector<Triangle>& triangles, std::vector<TriangleSOA<4>>& triangles4);
		bool intersect(const Scene& scene, const Ray& ray, Intersection& intersection) const;

	private:

		BVHNodeSOA<4>* nodesPtr = nullptr;

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
		}
	};
}
