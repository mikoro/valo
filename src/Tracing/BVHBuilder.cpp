// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Tracing/BVHBuilder.h"
#include "Tracing/BVH.h"
#include "Tracing/Triangle.h"
#include "Utils/Random.h"
#include "Utils/Timer.h"
#include "Math/Vector3.h"

using namespace Raycer;

BVHBuilder::BVHBuilder() : processedTrianglesCount(0)
{
}

void BVHBuilder::build(std::vector<Triangle>& triangles, const BVHBuildInfo& buildInfo, BVH& bvh, std::atomic<bool>& interrupted)
{
	Timer timer;
	Random random;

	BVHBuildEntry stack[128];
	bvh.nodes.clear();
	processedTrianglesCount = 0;

	uint64_t stackptr = 0;
	uint64_t nodeCount = 0;
	uint64_t leafCount = 0;

	enum { ROOT = -4, UNVISITED = -3, VISITED_TWICE = -1 };

	// push to stack
	stack[stackptr].start = 0;
	stack[stackptr].end = triangles.size();
	stack[stackptr].parent = ROOT;
	stackptr++;

	while (stackptr > 0)
	{
		if (interrupted)
			return;

		stackptr--;
		nodeCount++;

		// pop from stack
		BVHNode node;
		BVHBuildEntry buildEntry = stack[stackptr];
		node.rightOffset = UNVISITED;
		node.startOffset = buildEntry.start;
		node.triangleCount = buildEntry.end - buildEntry.start;
		node.splitAxis = 0;
		node.leftEnabled = 1;
		node.rightEnabled = 1;

		for (uint64_t i = buildEntry.start; i < buildEntry.end; ++i)
			node.aabb.expand(triangles[i].getAABB());

		// leaf node indicated by rightOffset == 0
		if (node.triangleCount <= buildInfo.maxLeafSize)
			node.rightOffset = 0;

		// update the parent rightOffset when visiting its right child
		if (buildEntry.parent != ROOT)
		{
			bvh.nodes[uint64_t(buildEntry.parent)].rightOffset++;

			if (bvh.nodes[uint64_t(buildEntry.parent)].rightOffset == VISITED_TWICE)
				bvh.nodes[uint64_t(buildEntry.parent)].rightOffset = int64_t(nodeCount) - 1 - buildEntry.parent;
		}

		// leaf node -> no further subdivision
		if (node.rightOffset == 0)
		{
			bvh.nodes.push_back(node);
			leafCount++;
			processedTrianglesCount += node.triangleCount;

			continue;
		}

		uint64_t splitIndex = 0;
		calculateSplit(triangles, node, splitIndex, buildEntry);
		bvh.nodes.push_back(node);

		// split failed -> fallback to middle split
		if (splitIndex <= buildEntry.start || splitIndex >= buildEntry.end)
			splitIndex = buildEntry.start + (buildEntry.end - buildEntry.start) / 2;

		// push right child
		stack[stackptr].start = splitIndex;
		stack[stackptr].end = buildEntry.end;
		stack[stackptr].parent = int64_t(nodeCount) - 1;
		stackptr++;

		// push left child
		stack[stackptr].start = buildEntry.start;
		stack[stackptr].end = splitIndex;
		stack[stackptr].parent = int64_t(nodeCount) - 1;
		stackptr++;
	}

	bvh.bvhHasBeenBuilt = true;
}

uint64_t BVHBuilder::getProcessedTrianglesCount() const
{
	return processedTrianglesCount;
}

void BVHBuilder::calculateSplit(std::vector<Triangle>& triangles, BVHNode& node, uint64_t& splitIndex, const BVHBuildEntry& buildEntry)
{
	double lowestScore = std::numeric_limits<double>::max();
	double parentSurfaceArea = node.aabb.getSurfaceArea();

	for (uint64_t axis = 0; axis <= 2; ++axis)
	{
		std::sort(triangles.begin() + buildEntry.start, triangles.begin() + buildEntry.end, [axis](const Triangle& t1, const Triangle& t2)
		{
			return t1.getAABB().getCenter().get(axis) < t2.getAABB().getCenter().get(axis);
		});

		AABB leftAABB;
		uint64_t leftCount = 0;

		for (uint64_t i = buildEntry.start; i < buildEntry.end; ++i)
		{
			leftAABB.expand(triangles[i].getAABB());
			leftCount++;

			AABB rightAABB;
			uint64_t rightCount = 0;

			for (uint64_t j = i + 1; j < buildEntry.end; ++j)
			{
				rightAABB.expand(triangles[j].getAABB());
				rightCount++;
			}

			double score = (leftAABB.getSurfaceArea() / parentSurfaceArea) * double(leftCount);
			score += (rightAABB.getSurfaceArea() / parentSurfaceArea) * double(rightCount);

			if (score < lowestScore)
			{
				node.splitAxis = axis;
				splitIndex = i + 1;
				lowestScore = score;
			}
		}
	}

	if (node.splitAxis != 2)
	{
		std::sort(triangles.begin() + buildEntry.start, triangles.begin() + buildEntry.end, [node](const Triangle& t1, const Triangle& t2)
		{
			return t1.getAABB().getCenter().get(node.splitAxis) < t2.getAABB().getCenter().get(node.splitAxis);
		});
	}
}
