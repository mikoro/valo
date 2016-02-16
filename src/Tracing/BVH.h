// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <vector>

#include "cereal/cereal.hpp"

#include "Tracing/AABB.h"
#include "Tracing/BVHBuilder.h"

namespace Raycer
{
	class Triangle;
	class Ray;
	class Intersection;
	
	class BVH
	{
	public:

		bool intersect(const std::vector<Triangle>& triangles, const Ray& ray, Intersection& intersection) const;

		void disableLeft();
		void disableRight();
		void revertDisable();

		bool bvhHasBeenBuilt = false;
		std::vector<BVHNode> nodes;

	private:

		uint64_t disableIndex = 0;
		std::vector<uint64_t> previousDisableIndices;

		friend class cereal::access;

		template <class Archive>
		void serialize(Archive& ar)
		{
			ar(CEREAL_NVP(bvhHasBeenBuilt),
				CEREAL_NVP(nodes));
		}
	};
}
