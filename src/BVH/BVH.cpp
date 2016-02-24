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

BVHSplitOutput BVH::calculateSplit(std::vector<Triangle*>& trianglePtrs, std::vector<BVHSplitCache>& cache, uint64_t start, uint64_t end)
{
	assert(end > start);

	BVHSplitOutput output;
	float lowestCost = std::numeric_limits<float>::max();
	AABB fullAABB[3];

	for (uint64_t axis = 0; axis <= 2; ++axis)
	{
		concurrency::parallel_sort(trianglePtrs.begin() + start, trianglePtrs.begin() + end, [axis](const Triangle* t1, const Triangle* t2)
		{
			return (&t1->center.x)[axis] < (&t2->center.x)[axis];
		});

		AABB rightAABB;
		uint64_t rightCount = 0;

		for (int64_t i = end - 1; i >= int64_t(start); --i)
		{
			rightAABB.expand(trianglePtrs[i]->aabb);
			rightCount++;

			cache[i].aabb = rightAABB;
			cache[i].cost = rightAABB.getSurfaceArea() * float(rightCount);
		}

		AABB leftAABB;
		uint64_t leftCount = 0;

		for (uint64_t i = start; i < end; ++i)
		{
			leftAABB.expand(trianglePtrs[i]->aabb);
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
		concurrency::parallel_sort(trianglePtrs.begin() + start, trianglePtrs.begin() + end, [output](const Triangle* t1, const Triangle* t2)
		{
			return (&t1->center.x)[output.axis] < (&t2->center.x)[output.axis];
		});
	}
	
	output.fullAABB = fullAABB[output.axis];

	return output;
}
