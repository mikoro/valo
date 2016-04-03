// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include <cfloat>

#ifdef _WIN32
#include <ppl.h>
#define PARALLEL_SORT concurrency::parallel_sort
#endif

#ifdef __linux
#include <parallel/algorithm>
#define PARALLEL_SORT __gnu_parallel::sort
#endif

#include "BVH/BVH.h"

using namespace Raycer;

void BVH::build(std::vector<Triangle>& triangles)
{
	switch (type)
	{
		case BVHType::BVH1: bvh1.build(triangles); break;
		case BVHType::BVH4: bvh4.build(triangles); break;
		case BVHType::BVH8: bvh8.build(triangles); break;
		default: break;
	}
}

CUDA_CALLABLE bool BVH::intersect(const Scene& scene, const Ray& ray, Intersection& intersection) const
{
	switch (type)
	{
		case BVHType::BVH1: return bvh1.intersect(scene, ray, intersection);
		case BVHType::BVH4: return bvh4.intersect(scene, ray, intersection);
		case BVHType::BVH8: return bvh8.intersect(scene, ray, intersection);
		default: return false;
	}
}

BVHSplitOutput BVH::calculateSplit(std::vector<BVHBuildTriangle>& buildTriangles, std::vector<BVHSplitCache>& cache, uint32_t start, uint32_t end)
{
	assert(end > start);

	BVHSplitOutput output;
	float lowestCost = FLT_MAX;
	AABB fullAABB[3];

	for (uint32_t axis = 0; axis <= 2; ++axis)
	{
		PARALLEL_SORT(buildTriangles.begin() + start, buildTriangles.begin() + end, [axis](const BVHBuildTriangle& t1, const BVHBuildTriangle& t2)
		{
			return (&t1.center.x)[axis] < (&t2.center.x)[axis];
		});

		AABB rightAABB;
		uint32_t rightCount = 0;

		for (int32_t i = end - 1; i >= int32_t(start); --i)
		{
			rightAABB.expand(buildTriangles[i].aabb);
			rightCount++;

			cache[i].aabb = rightAABB;
			cache[i].cost = rightAABB.getSurfaceArea() * float(rightCount);
		}

		AABB leftAABB;
		uint32_t leftCount = 0;

		for (uint32_t i = start; i < end; ++i)
		{
			leftAABB.expand(buildTriangles[i].aabb);
			leftCount++;

			float cost = leftAABB.getSurfaceArea() * float(leftCount);

			if (i + 1 < end)
				cost += cache[i + 1].cost;

			if (cost < lowestCost)
			{
				output.index = i + 1;
				output.axis = axis;
				output.leftAABB = leftAABB;

				if (output.index < end)
					output.rightAABB = cache[output.index].aabb;

				lowestCost = cost;
			}
		}

		fullAABB[axis] = leftAABB;
	}

	assert(output.index >= start && output.index <= end);

	if (output.axis != 2)
	{
		PARALLEL_SORT(buildTriangles.begin() + start, buildTriangles.begin() + end, [output](const BVHBuildTriangle& t1, const BVHBuildTriangle& t2)
		{
			return (&t1.center.x)[output.axis] < (&t2.center.x)[output.axis];
		});
	}
	
	output.fullAABB = fullAABB[output.axis];
	return output;
}
