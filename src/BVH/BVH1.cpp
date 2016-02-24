// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "BVH/BVH1.h"
#include "App.h"
#include "Utils/Log.h"
#include "Utils/Timer.h"
#include "Tracing/Triangle.h"
#include "Tracing/Ray.h"
#include "Tracing/Intersection.h"

using namespace Raycer;

namespace
{
	struct BVH1BuildEntry
	{
		uint64_t start;
		uint64_t end;
		int64_t parent;
	};
}

void BVH1::build(std::vector<Triangle>& triangles, uint64_t maxLeafSize)
{
	Log& log = App::getLog();

	log.logInfo("BVH1 building started (triangles: %d)", triangles.size());

	Timer timer;
	uint64_t triangleCount = triangles.size();
	//std::array<std::vector<Triangle*>, 3> trianglePtrs;
	std::vector<Triangle*> trianglePtrs;
	std::vector<BVHSplitCache> cache(triangleCount);
	BVHSplitInput splitInput;
	BVHSplitOutput splitOutput;

	splitInput.trianglePtrs = &trianglePtrs;
	splitInput.cache = &cache;

	//sortTriangles(triangles, trianglePtrs);

	for (Triangle& triangle : triangles)
		trianglePtrs.push_back(&triangle);

	nodes.clear();
	nodes.reserve(triangleCount);

	enum { ROOT = -4, UNVISITED = -3, VISITED_TWICE = -1 };

	BVH1BuildEntry stack[128];
	uint64_t stackIndex = 0;
	uint64_t nodeCount = 0;
	uint64_t leafCount = 0;

	// push to stack
	stack[stackIndex].start = 0;
	stack[stackIndex].end = triangleCount;
	stack[stackIndex].parent = ROOT;
	stackIndex++;

	while (stackIndex > 0)
	{
		nodeCount++;

		// pop from stack
		BVH1Node node;
		BVH1BuildEntry buildEntry = stack[--stackIndex];
		node.rightOffset = UNVISITED;
		node.startOffset = uint32_t(buildEntry.start);
		node.triangleCount = uint32_t(buildEntry.end - buildEntry.start);
		node.splitAxis = 0;

		// leaf node indicated by rightOffset == 0
		if (node.triangleCount <= maxLeafSize)
			node.rightOffset = 0;

		// update the parent rightOffset when visiting its right child
		if (buildEntry.parent != ROOT)
		{
			uint64_t parent = uint64_t(buildEntry.parent);

			nodes[parent].rightOffset++;

			if (nodes[parent].rightOffset == VISITED_TWICE)
				nodes[parent].rightOffset = int32_t(nodeCount - 1 - parent);
		}

		if (node.rightOffset != 0)
		{
			splitInput.start = buildEntry.start;
			splitInput.end = buildEntry.end;

			splitOutput = calculateSplit(splitInput);

			node.splitAxis = uint32_t(splitOutput.axis);
			node.aabb = splitOutput.fullAABB;
		}

		nodes.push_back(node);

		if (node.rightOffset == 0)
		{
			leafCount++;
			continue;
		}

		// push right child
		stack[stackIndex].start = splitOutput.index;
		stack[stackIndex].end = buildEntry.end;
		stack[stackIndex].parent = int64_t(nodeCount) - 1;
		stackIndex++;

		// push left child
		stack[stackIndex].start = buildEntry.start;
		stack[stackIndex].end = splitOutput.index;
		stack[stackIndex].parent = int64_t(nodeCount) - 1;
		stackIndex++;
	}

	nodes.shrink_to_fit();

	std::vector<Triangle> tempTriangles(triangleCount);

	for (uint64_t i = 0; i < triangleCount; ++i)
		tempTriangles[i] = *trianglePtrs[i];

	triangles = tempTriangles;

	log.logInfo("BVH1 building finished (time: %s, nodes: %d, leafs: %d)", timer.getElapsed().getString(true), nodeCount - leafCount, leafCount);
}

bool BVH1::intersect(const std::vector<Triangle>& triangles, const Ray& ray, Intersection& intersection) const
{
	if (ray.fastOcclusion && intersection.wasFound)
		return true;

	uint64_t stack[64];
	uint64_t stackIndex = 0;
	bool wasFound = false;

	stack[stackIndex++] = 0;

	while (stackIndex > 0)
	{
		uint64_t nodeIndex = stack[--stackIndex];
		const BVH1Node& node = nodes[nodeIndex];

		// leaf node
		if (node.rightOffset == 0)
		{
			for (uint64_t i = 0; i < node.triangleCount; ++i)
			{
				if (triangles[node.startOffset + i].intersect(ray, intersection))
				{
					if (ray.fastOcclusion)
						return true;

					wasFound = true;
				}
			}

			continue;
		}

		if (node.aabb.intersects(ray))
		{
			if (ray.directionIsNegative[node.splitAxis])
			{
				stack[stackIndex++] = nodeIndex + 1; // left child
				stack[stackIndex++] = nodeIndex + uint64_t(node.rightOffset); // right child
			}
			else
			{
				stack[stackIndex++] = nodeIndex + uint64_t(node.rightOffset); // right child
				stack[stackIndex++] = nodeIndex + 1; // left child
			}
		}
	}

	return wasFound;
}
