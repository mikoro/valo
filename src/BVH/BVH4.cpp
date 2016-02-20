// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "BVH/BVH4.h"
#include "App.h"
#include "Utils/Log.h"
#include "Utils/Timer.h"
#include "Tracing/Triangle.h"
#include "Tracing/Ray.h"
#include "Tracing/Intersection.h"

using namespace Raycer;

namespace
{
	struct BVH4BuildEntry
	{
		uint64_t start;
		uint64_t end;
		int64_t parent;
		int64_t child;
	};
}

void BVH4::build(std::vector<Triangle>& triangles, const BVHBuildInfo& buildInfo)
{
	Log& log = App::getLog();

	log.logInfo("BVH4 building started (triangles: %d)", triangles.size());

	Timer timer;
	uint64_t triangleCount = triangles.size();
	uint64_t failedLeftSplitCount = 0;
	uint64_t failedMiddleSplitCount = 0;
	uint64_t failedRightSplitCount = 0;
	std::vector<Triangle*> trianglePtrs;
	std::vector<float> rightScores(triangleCount);
	BVHSplitInput splitInput;
	BVHSplitOutput splitOutputs[3];

	splitInput.trianglePtrs = &trianglePtrs;
	splitInput.rightScores = &rightScores;
	splitInput.parentSurfaceArea = 1.0f;

	nodes.clear();
	nodes.reserve(triangleCount);
	trianglePtrs.reserve(triangleCount);

	for (Triangle& triangle : triangles)
		trianglePtrs.push_back(&triangle);

	BVH4BuildEntry stack[128];
	uint64_t stackIndex = 0;
	uint64_t nodeCount = 0;
	uint64_t leafCount = 0;

	// push to stack
	stack[stackIndex].start = 0;
	stack[stackIndex].end = triangleCount;
	stack[stackIndex].parent = -1;
	stackIndex++;

	while (stackIndex > 0)
	{
		stackIndex--;
		nodeCount++;

		// pop from stack
		BVH4BuildEntry buildEntry = stack[stackIndex];
		BVH4Node node;
		node.triangleOffset = buildEntry.start;
		node.triangleCount = buildEntry.end - buildEntry.start;
		node.isLeaf = (node.triangleCount <= buildInfo.maxLeafSize);

		if (buildEntry.parent != -1 && buildEntry.child != -1)
		{
			uint64_t parent = uint64_t(buildEntry.parent);
			uint64_t child = uint64_t(buildEntry.child);

			nodes[parent].rightOffset[child] = nodeCount - 1 - parent;
		}

		if (!node.isLeaf)
		{
			// middle split
			splitInput.start = buildEntry.start;
			splitInput.end = buildEntry.end;
			splitOutputs[1] = calculateSplit(splitInput);

			if (splitOutputs[1].index - buildEntry.start < 2 || buildEntry.end - splitOutputs[1].index < 2)
			{
				node.isLeaf = true;
				failedMiddleSplitCount++;
			}

			if (!node.isLeaf)
			{
				// left split
				splitInput.start = buildEntry.start;
				splitInput.end = splitOutputs[1].index;
				splitOutputs[0] = calculateSplit(splitInput);

				// right split
				splitInput.start = splitOutputs[1].index;
				splitInput.end = buildEntry.end;
				splitOutputs[2] = calculateSplit(splitInput);

				if (splitOutputs[0].failed)
					failedLeftSplitCount++;

				if (splitOutputs[2].failed)
					failedRightSplitCount++;

				node.aabb[0] = splitOutputs[0].leftAABB;
				node.aabb[1] = splitOutputs[0].rightAABB;
				node.aabb[2] = splitOutputs[2].leftAABB;
				node.aabb[3] = splitOutputs[2].rightAABB;
			}
		}

		nodes.push_back(node);

		if (node.isLeaf)
		{
			leafCount++;
			continue;
		}

		// push right child 2
		stack[stackIndex].start = splitOutputs[2].index;
		stack[stackIndex].end = buildEntry.end;
		stack[stackIndex].parent = int64_t(nodeCount) - 1;
		stack[stackIndex].child = 2;
		stackIndex++;

		// push right child 1
		stack[stackIndex].start = splitOutputs[1].index;
		stack[stackIndex].end = splitOutputs[2].index;
		stack[stackIndex].parent = int64_t(nodeCount) - 1;
		stack[stackIndex].child = 1;
		stackIndex++;

		// push right child 0
		stack[stackIndex].start = splitOutputs[0].index;
		stack[stackIndex].end = splitOutputs[1].index;
		stack[stackIndex].parent = int64_t(nodeCount) - 1;
		stack[stackIndex].child = 0;
		stackIndex++;

		// push left child
		stack[stackIndex].start = buildEntry.start;
		stack[stackIndex].end = splitOutputs[0].index;
		stack[stackIndex].parent = int64_t(nodeCount) - 1;
		stack[stackIndex].child = -1;
		stackIndex++;
	}

	built = true;
	nodes.shrink_to_fit();

	std::vector<Triangle> tempTriangles(triangleCount);

	for (uint64_t i = 0; i < triangleCount; ++i)
		tempTriangles[i] = *trianglePtrs[i];

	triangles = tempTriangles;

	log.logInfo("BVH4 building finished (time: %s, nodes: %d, leafs: %d, failed splits: (%d, %d, %d), triangles/leaf: %.2f)", timer.getElapsed().getString(true), nodeCount - leafCount, leafCount, failedLeftSplitCount, failedMiddleSplitCount, failedRightSplitCount, float(triangleCount) / float(leafCount));
}

bool BVH4::intersect(const std::vector<Triangle>& triangles, const Ray& ray, Intersection& intersection) const
{
	if (ray.fastOcclusion && intersection.wasFound)
		return true;

	uint64_t stack[64];
	uint64_t stackIndex = 0;
	bool wasFound = false;

	// push to stack
	stack[stackIndex] = 0;
	stackIndex++;

	while (stackIndex > 0)
	{
		// pop from stack
		stackIndex--;
		uint64_t nodeIndex = stack[stackIndex];
		const BVH4Node& node = nodes[nodeIndex];

		if (node.isLeaf)
		{
			for (uint64_t i = 0; i < node.triangleCount; ++i)
			{
				if (triangles[node.triangleOffset + i].intersect(ray, intersection))
				{
					if (ray.fastOcclusion)
						return true;

					wasFound = true;
				}
			}

			continue;
		}

		std::array<bool, 4> intersects = AABB::intersects(node, ray);

		if (intersects[3])
		{
			stack[stackIndex] = nodeIndex + node.rightOffset[2];
			stackIndex++;
		}

		if (intersects[2])
		{
			stack[stackIndex] = nodeIndex + node.rightOffset[1];
			stackIndex++;
		}

		if (intersects[1])
		{
			stack[stackIndex] = nodeIndex + node.rightOffset[0];
			stackIndex++;
		}

		if (intersects[0])
		{
			stack[stackIndex] = nodeIndex + 1;
			stackIndex++;
		}
	}

	return wasFound;
}
