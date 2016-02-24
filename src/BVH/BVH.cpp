// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "BVH/BVH.h"
#include "BVH/BVH1.h"
#include "BVH/BVH4.h"
#include "Tracing/Triangle.h"

using namespace Raycer;

std::unique_ptr<BVH> BVH::getBVH(BVHType type)
{
	switch (type)
	{
		case BVHType::BVH1: return std::make_unique<BVH1>();
		case BVHType::BVH4: return std::make_unique<BVH4>();
		//case BVHType::BVH8: return std::make_unique<BVH1>();
		//case BVHType::SBVH1: return std::make_unique<BVH1>();
		default: throw std::runtime_error("Unknown BVH type");
	}
}

void BVH::sortTriangles(std::vector<Triangle>& triangles, std::array<std::vector<Triangle*>, 3>& sortedTrianglePtrs)
{
	uint64_t triangleCount = triangles.size();

	sortedTrianglePtrs[0].resize(triangleCount);

	for (uint64_t i = 0; i < triangleCount; ++i)
		sortedTrianglePtrs[0][i] = &triangles[i];

	sortedTrianglePtrs[1] = sortedTrianglePtrs[0];
	sortedTrianglePtrs[2] = sortedTrianglePtrs[0];

	for (uint64_t i = 0; i < 3; ++i)
	{
		concurrency::parallel_sort(sortedTrianglePtrs[i].begin(), sortedTrianglePtrs[i].end(), [i](const Triangle* t1, const Triangle* t2)
		{
			return (&t1->center.x)[i] < (&t2->center.x)[i];
		});
	}
}

BVHSplitOutput BVH::calculateSplit(std::array<std::vector<Triangle*>, 3>& sortedTrianglePtrs, std::vector<BVHSplitCache>& cache, uint64_t start, uint64_t end)
{
	assert(end > start);

	BVHSplitOutput output;
	float lowestCost = std::numeric_limits<float>::max();
	AABB fullAABB[3];

	for (uint64_t axis = 0; axis <= 2; ++axis)
	{
		AABB rightAABB;
		uint64_t rightCount = 0;

		for (int64_t i = end - 1; i >= int64_t(start); --i)
		{
			rightAABB.expand(sortedTrianglePtrs[axis][i]->aabb);
			rightCount++;

			cache[i].aabb = rightAABB;
			cache[i].cost = rightAABB.getSurfaceArea() * float(rightCount);
		}

		AABB leftAABB;
		uint64_t leftCount = 0;

		for (uint64_t i = start; i < end; ++i)
		{
			leftAABB.expand(sortedTrianglePtrs[axis][i]->aabb);
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

	output.fullAABB = fullAABB[output.axis];

	assert(output.index >= start && output.index <= end);

	return output;
}
