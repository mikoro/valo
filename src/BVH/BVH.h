// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <memory>
#include <vector>

#include "Tracing/AABB.h"

namespace Raycer
{
	class Triangle;
	class Ray;
	class Intersection;

	enum class BVHType { BVH1, BVH4, BVH8, SBVH1 };

	struct BVHSplitCache
	{
		AABB aabb;
		float cost;
	};

	struct BVHSplitOutput
	{
		uint64_t index;
		uint64_t axis;
		AABB fullAABB;
		AABB leftAABB;
		AABB rightAABB;
	};

	class BVH
	{
	public:

		virtual ~BVH() {}

		virtual void build(std::vector<Triangle>& triangles, uint64_t maxLeafSize) = 0;
		virtual bool intersect(const std::vector<Triangle>& triangles, const Ray& ray, Intersection& intersection) const = 0;

		static std::unique_ptr<BVH> getBVH(BVHType type);

	protected:

		static BVHSplitOutput calculateSplit(std::vector<Triangle*>& trianglePtrs, std::vector<BVHSplitCache>& cache, uint64_t start, uint64_t end);
	};
}
