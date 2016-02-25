// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#pragma once

#include <memory>
#include <vector>

#include "Tracing/Aabb.h"

namespace Raycer
{
	class Scene;
	class Triangle;
	class Ray;
	class Intersection;

	enum class BVHType { BVH1, BVH4 };

	struct BVHBuildTriangle
	{
		Triangle* triangle;
		Aabb aabb;
		Vector3 center;
	};

	struct BVHSplitCache
	{
		Aabb aabb;
		float cost;
	};

	struct BVHSplitOutput
	{
		uint64_t index;
		uint64_t axis;
		Aabb fullAabb;
		Aabb leftAabb;
		Aabb rightAabb;
	};

	using BVHBuildTriangleVector = std::vector<BVHBuildTriangle, boost::alignment::aligned_allocator<BVHBuildTriangle, 16>>;

	class BVH
	{
	public:

		virtual ~BVH() {}

		virtual void build(std::vector<Triangle>& triangles, uint64_t maxLeafSize) = 0;
		virtual bool intersect(const Scene& scene, const Ray& ray, Intersection& intersection) const = 0;

		static std::unique_ptr<BVH> getBVH(BVHType type);

	protected:

		static BVHSplitOutput calculateSplit(BVHBuildTriangleVector& buildTriangles, std::vector<BVHSplitCache>& cache, uint64_t start, uint64_t end);
	};
}
