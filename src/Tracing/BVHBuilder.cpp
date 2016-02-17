// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include <ppl.h>

#include "Tracing/BVHBuilder.h"
#include "Tracing/BVH.h"
#include "Tracing/Triangle.h"
#include "App.h"
#include "Utils/Log.h"
#include "Utils/Timer.h"
#include "Math/Vector3.h"

using namespace Raycer;

void BVHBuilder::build(std::vector<Triangle>& triangles, const BVHBuildInfo& buildInfo, BVH& bvh)
{
	Log& log = App::getLog();

	log.logInfo("BVH building started (triangles: %d)", triangles.size());

	Timer timer;
	BVHBuildEntry stack[128];
	uint64_t triangleCount = triangles.size();
	std::vector<float> rightScores(triangleCount);
	std::vector<Triangle*> trianglePtrs;
	uint64_t failedSplitCount = 0;

	bvh.nodes.clear();
	trianglePtrs.reserve(triangleCount);

	for (Triangle& triangle : triangles)
		trianglePtrs.push_back(&triangle);
	
	uint64_t stackptr = 0;
	uint64_t nodeCount = 0;
	uint64_t leafCount = 0;

	enum { ROOT = -4, UNVISITED = -3, VISITED_TWICE = -1 };

	// push to stack
	stack[stackptr].start = 0;
	stack[stackptr].end = triangleCount;
	stack[stackptr].parent = ROOT;
	stackptr++;

	while (stackptr > 0)
	{
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
			node.aabb.expand(trianglePtrs[i]->aabb);

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
			continue;
		}

		uint64_t splitIndex = 0;
		calculateSplit(trianglePtrs, node, splitIndex, buildEntry, rightScores);
		bvh.nodes.push_back(node);

		// split failed -> fallback to middle split
		if (splitIndex <= buildEntry.start || splitIndex >= buildEntry.end)
		{
			splitIndex = buildEntry.start + (buildEntry.end - buildEntry.start) / 2;
			failedSplitCount++;
		}

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

	std::vector<Triangle> tempTriangles(triangleCount);
	
	for (uint64_t i = 0; i < triangleCount; ++i)
		tempTriangles[i] = *trianglePtrs[i];
	
	triangles = tempTriangles;

	log.logInfo("BVH building finished (time: %s, nodes: %d, leafs: %d, failed splits: %d)", timer.getElapsed().getString(true), nodeCount, leafCount, failedSplitCount);
}

void BVHBuilder::calculateSplit(std::vector<Triangle*>& trianglePtrs, BVHNode& node, uint64_t& splitIndex, const BVHBuildEntry& buildEntry, std::vector<float>& rightScores)
{
	float lowestScore = std::numeric_limits<float>::max();
	float parentSurfaceArea = node.aabb.getSurfaceArea();
	
	for (uint64_t axis = 0; axis <= 2; ++axis)
	{
		concurrency::parallel_sort(trianglePtrs.begin() + buildEntry.start, trianglePtrs.begin() + buildEntry.end, [axis](const Triangle* t1, const Triangle* t2)
		{
			return (&t1->center.x)[axis] < (&t2->center.x)[axis];
		});

		AABB rightAABB;
		uint64_t rightCount = 0;

		for (int64_t i = buildEntry.end - 1; i >= int64_t(buildEntry.start); --i)
		{
			rightAABB.expand(trianglePtrs[i]->aabb);
			rightCount++;

			rightScores[i] = (rightAABB.getSurfaceArea() / parentSurfaceArea) * float(rightCount);
		}

		AABB leftAABB;
		uint64_t leftCount = 0;

		for (uint64_t i = buildEntry.start; i < buildEntry.end - 1; ++i)
		{
			leftAABB.expand(trianglePtrs[i]->aabb);
			leftCount++;

			float score = (leftAABB.getSurfaceArea() / parentSurfaceArea) * float(leftCount) + rightScores[i + 1];

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
		concurrency::parallel_sort(trianglePtrs.begin() + buildEntry.start, trianglePtrs.begin() + buildEntry.end, [node](const Triangle* t1, const Triangle* t2)
		{
			return (&t1->center.x)[node.splitAxis] < (&t2->center.x)[node.splitAxis];
		});
	}
}
