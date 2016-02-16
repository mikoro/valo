// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Tracing/BVH.h"
#include "Tracing/Ray.h"
#include "Tracing/Intersection.h"
#include "Tracing/Triangle.h"
#include "App.h"
#include "Utils/Log.h"
#include "Utils/Random.h"
#include "Utils/Timer.h"
#include "Math/Vector3.h"

using namespace Raycer;

bool BVH::intersect(const std::vector<Triangle>& triangles, const Ray& ray, Intersection& intersection) const
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
		const BVHNode& node = nodes[nodeIndex];

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

void BVH::build(std::vector<Triangle>& triangles, const BVHBuildInfo& buildInfo)
{
	Log& log = App::getLog();

	log.logInfo("Building BVH (triangles: %d)", triangles.size());

	Timer timer;
	Random random;

	BVHBuildEntry stack[128];
	nodes.clear();

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

		uint64_t splitIndex = 0;
		calculateSplit(triangles, node, splitIndex, buildEntry);
		nodes.push_back(node);

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

	bvhHasBeenBuilt = true;

	log.logInfo("BVH building finished (time: %.2f ms, nodes: %d, leafs: %d)", timer.getElapsedMilliseconds(), nodeCount, leafCount);
}

bool BVH::hasBeenBuilt() const
{
	return bvhHasBeenBuilt;
}

void BVH::disableLeft()
{
	if (nodes[disableIndex].rightOffset == 0)
		return;

	previousDisableIndices.push_back(disableIndex);
	nodes[disableIndex].leftEnabled = 0;
	disableIndex += nodes[disableIndex].rightOffset;
}

void BVH::disableRight()
{
	if (nodes[disableIndex].rightOffset == 0)
		return;

	previousDisableIndices.push_back(disableIndex);
	nodes[disableIndex].rightEnabled = 0;
	++disableIndex;
}

void BVH::revertDisable()
{
	if (previousDisableIndices.size() == 0)
		return;

	disableIndex = previousDisableIndices.back();
	previousDisableIndices.pop_back();

	nodes[disableIndex].leftEnabled = 1;
	nodes[disableIndex].rightEnabled = 1;
}

void BVH::calculateSplit(std::vector<Triangle>& triangles, BVHNode& node, uint64_t& splitIndex, const BVHBuildEntry& buildEntry)
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
