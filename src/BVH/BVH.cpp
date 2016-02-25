// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "BVH/BVH.h"
#include "BVH/BVH1.h"
#include "BVH/BVH4.h"

using namespace Raycer;

std::unique_ptr<BVH> BVH::getBVH(BVHType type)
{
	switch (type)
	{
		case BVHType::BVH1: return std::make_unique<BVH1>();
		case BVHType::BVH4: return std::make_unique<BVH4>();
		default: throw std::runtime_error("Unknown BVH type");
	}
}

BVHSplitOutput BVH::calculateSplit(std::vector<BVHBuildTriangle>& buildTriangles, std::vector<BVHSplitCache>& cache, uint64_t start, uint64_t end)
{
	assert(end > start);

	BVHSplitOutput output;
	float lowestCost = std::numeric_limits<float>::max();
	AABB fullAABB[3];

	for (uint64_t axis = 0; axis <= 2; ++axis)
	{
		concurrency::parallel_sort(buildTriangles.begin() + start, buildTriangles.begin() + end, [axis](const BVHBuildTriangle& t1, const BVHBuildTriangle& t2)
		{
			return (&t1.center.x)[axis] < (&t2.center.x)[axis];
		});

		AABB rightAABB;
		uint64_t rightCount = 0;

		for (int64_t i = end - 1; i >= int64_t(start); --i)
		{
			rightAABB.expand(buildTriangles[i].aabb);
			rightCount++;

			cache[i].aabb = rightAABB;
			cache[i].cost = rightAABB.getSurfaceArea() * float(rightCount);
		}

		AABB leftAABB;
		uint64_t leftCount = 0;

		for (uint64_t i = start; i < end; ++i)
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
		concurrency::parallel_sort(buildTriangles.begin() + start, buildTriangles.begin() + end, [output](const BVHBuildTriangle& t1, const BVHBuildTriangle& t2)
		{
			return (&t1.center.x)[output.axis] < (&t2.center.x)[output.axis];
		});
	}
	
	output.fullAABB = fullAABB[output.axis];

	return output;
}
