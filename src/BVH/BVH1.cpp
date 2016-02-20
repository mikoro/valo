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

void BVH1::build(std::vector<Triangle>& triangles, const BVHBuildInfo& buildInfo)
{
	Log& log = App::getLog();

	log.logInfo("BVH1 building started (triangles: %d)", triangles.size());

	Timer timer;
	BVHBuildEntry stack[128];
	uint64_t triangleCount = triangles.size();
	uint64_t failedSplitCount = 0;
	std::vector<Triangle*> trianglePtrs;
	std::vector<float> rightScores(triangleCount);
	BVHSplitInput splitInput;
	splitInput.trianglePtrs = &trianglePtrs;
	splitInput.rightScores = &rightScores;

	nodes.clear();
	nodes.reserve(triangleCount);
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
		BVH1Node node;
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
			nodes[uint64_t(buildEntry.parent)].rightOffset++;

			if (nodes[uint64_t(buildEntry.parent)].rightOffset == VISITED_TWICE)
				nodes[uint64_t(buildEntry.parent)].rightOffset = int64_t(nodeCount) - 1 - buildEntry.parent;
		}

		// leaf node -> no further subdivision
		if (node.rightOffset == 0)
		{
			nodes.push_back(node);
			leafCount++;
			continue;
		}

		splitInput.startIndex = buildEntry.start;
		splitInput.endIndex = buildEntry.end;
		splitInput.nodeSurfaceArea = node.aabb.getSurfaceArea();
		BVHSplitOutput splitOutput = calculateSplit(splitInput);

		node.splitAxis = splitOutput.splitAxis;
		nodes.push_back(node);

		// split failed -> fallback to middle split
		if (splitOutput.splitIndex <= buildEntry.start || splitOutput.splitIndex >= buildEntry.end)
		{
			splitOutput.splitIndex = buildEntry.start + (buildEntry.end - buildEntry.start) / 2;
			failedSplitCount++;
		}

		// push right child
		stack[stackptr].start = splitOutput.splitIndex;
		stack[stackptr].end = buildEntry.end;
		stack[stackptr].parent = int64_t(nodeCount) - 1;
		stackptr++;

		// push left child
		stack[stackptr].start = buildEntry.start;
		stack[stackptr].end = splitOutput.splitIndex;
		stack[stackptr].parent = int64_t(nodeCount) - 1;
		stackptr++;
	}

	built = true;
	nodes.shrink_to_fit();

	std::vector<Triangle> tempTriangles(triangleCount);

	for (uint64_t i = 0; i < triangleCount; ++i)
		tempTriangles[i] = *trianglePtrs[i];

	triangles = tempTriangles;

	log.logInfo("BVH1 building finished (time: %s, nodes: %d, leafs: %d, failed splits: %d)", timer.getElapsed().getString(true), nodeCount - leafCount, leafCount, failedSplitCount);
}

bool BVH1::intersect(const std::vector<Triangle>& triangles, const Ray& ray, Intersection& intersection) const
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
		const BVH1Node& node = nodes[nodeIndex];

		if (node.aabb.intersects(ray))
		{
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
			}
			else // travel down the tree
			{
				if (ray.directionIsNegative[node.splitAxis])
				{
					// seems to perform better like this (inverted logic?)

					if (node.leftEnabled)
					{
						// left child
						stack[stackIndex] = nodeIndex + 1;
						stackIndex++;
					}

					if (node.rightEnabled)
					{
						// right child
						stack[stackIndex] = nodeIndex + uint64_t(node.rightOffset);
						stackIndex++;
					}
				}
				else
				{
					if (node.rightEnabled)
					{
						// right child
						stack[stackIndex] = nodeIndex + uint64_t(node.rightOffset);
						stackIndex++;
					}

					if (node.leftEnabled)
					{
						// left child
						stack[stackIndex] = nodeIndex + 1;
						stackIndex++;
					}
				}
			}
		}
	}

	return wasFound;
}

void BVH1::disableLeft()
{
	if (nodes[disableIndex].rightOffset == 0)
		return;

	previousDisableIndices.push_back(disableIndex);
	nodes[disableIndex].leftEnabled = 0;
	disableIndex += nodes[disableIndex].rightOffset;
}

void BVH1::disableRight()
{
	if (nodes[disableIndex].rightOffset == 0)
		return;

	previousDisableIndices.push_back(disableIndex);
	nodes[disableIndex].rightEnabled = 0;
	++disableIndex;
}

void BVH1::undoDisable()
{
	if (previousDisableIndices.size() == 0)
		return;

	disableIndex = previousDisableIndices.back();
	previousDisableIndices.pop_back();

	nodes[disableIndex].leftEnabled = 1;
	nodes[disableIndex].rightEnabled = 1;
}
