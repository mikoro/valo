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
	Aabb fullAabb[3];

	for (uint64_t axis = 0; axis <= 2; ++axis)
	{
		concurrency::parallel_sort(buildTriangles.begin() + start, buildTriangles.begin() + end, [axis](const BVHBuildTriangle& t1, const BVHBuildTriangle& t2)
		{
			return (&t1.center.x)[axis] < (&t2.center.x)[axis];
		});

		Aabb rightAabb;
		uint64_t rightCount = 0;

		for (int64_t i = end - 1; i >= int64_t(start); --i)
		{
			rightAabb.expand(buildTriangles[i].aabb);
			rightCount++;

			cache[i].aabb = rightAabb;
			cache[i].cost = rightAabb.getSurfaceArea() * float(rightCount);
		}

		Aabb leftAabb;
		uint64_t leftCount = 0;

		for (uint64_t i = start; i < end; ++i)
		{
			leftAabb.expand(buildTriangles[i].aabb);
			leftCount++;

			float cost = leftAabb.getSurfaceArea() * float(leftCount);

			if (i + 1 < end)
				cost += cache[i + 1].cost;

			if (cost < lowestCost)
			{
				output.index = i + 1;
				output.axis = axis;
				output.leftAabb = leftAabb;

				if (output.index < end)
					output.rightAabb = cache[output.index].aabb;

				lowestCost = cost;
			}
		}

		fullAabb[axis] = leftAabb;
	}

	assert(output.index >= start && output.index <= end);

	if (output.axis != 2)
	{
		concurrency::parallel_sort(buildTriangles.begin() + start, buildTriangles.begin() + end, [output](const BVHBuildTriangle& t1, const BVHBuildTriangle& t2)
		{
			return (&t1.center.x)[output.axis] < (&t2.center.x)[output.axis];
		});
	}
	
	output.fullAabb = fullAabb[output.axis];

	return output;
}
