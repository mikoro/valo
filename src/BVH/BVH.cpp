// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include <ppl.h>

#include "BVH/BVH.h"
#include "BVH/BVH1.h"
#include "Tracing/Triangle.h"

using namespace Raycer;

std::unique_ptr<BVH> BVH::getBVH(BVHType type)
{
	switch (type)
	{
		case BVHType::BVH1: return std::make_unique<BVH1>();
		case BVHType::BVH4: return std::make_unique<BVH1>();
		case BVHType::BVH8: return std::make_unique<BVH1>();
		case BVHType::SBVH1: return std::make_unique<BVH1>();
		default: throw std::runtime_error("Unknown BVH type");
	}
}

bool BVH::hasBeenBuilt()
{
	return built;
}

void BVH::disableLeft()
{	
}

void BVH::disableRight()
{
}

void BVH::undoDisable()
{
}

BVHSplitOutput BVH::calculateSplit(const BVHSplitInput& input)
{
	assert(endIndex > startIndex);

	BVHSplitOutput output;
	float lowestScore = std::numeric_limits<float>::max();

	for (uint64_t axis = 0; axis <= 2; ++axis)
	{
		concurrency::parallel_sort(input.trianglePtrs->begin() + input.startIndex, input.trianglePtrs->begin() + input.endIndex, [axis](const Triangle* t1, const Triangle* t2)
		{
			return (&t1->center.x)[axis] < (&t2->center.x)[axis];
		});

		AABB rightAABB;
		uint64_t rightCount = 0;

		for (int64_t i = input.endIndex - 1; i >= int64_t(input.startIndex); --i)
		{
			rightAABB.expand((*input.trianglePtrs)[i]->aabb);
			rightCount++;

			(*input.rightScores)[i] = (rightAABB.getSurfaceArea() / input.nodeSurfaceArea) * float(rightCount);
		}

		AABB leftAABB;
		uint64_t leftCount = 0;

		for (uint64_t i = input.startIndex; i < input.endIndex - 1; ++i)
		{
			leftAABB.expand((*input.trianglePtrs)[i]->aabb);
			leftCount++;

			float score = (leftAABB.getSurfaceArea() / input.nodeSurfaceArea) * float(leftCount) + (*input.rightScores)[i + 1];

			if (score < lowestScore)
			{
				output.splitAxis = axis;
				output.splitIndex = i + 1;
				lowestScore = score;
			}
		}
	}

	if (output.splitAxis != 2)
	{
		concurrency::parallel_sort(input.trianglePtrs->begin() + input.startIndex, input.trianglePtrs->begin() + input.endIndex, [output](const Triangle* t1, const Triangle* t2)
		{
			return (&t1->center.x)[output.splitAxis] < (&t2->center.x)[output.splitAxis];
		});
	}

	return output;
}
