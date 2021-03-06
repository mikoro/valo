﻿// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "BVH/BVH2.h"
#include "App.h"
#include "Core/Common.h"
#include "Utils/Log.h"
#include "Utils/Timer.h"
#include "Core/Scene.h"
#include "Core/Triangle.h"
#include "Core/Ray.h"
#include "Core/Intersection.h"

using namespace Valo;

namespace
{
	struct BVH2BuildEntry
	{
		uint32_t start;
		uint32_t end;
		int32_t parent;
	};
}

BVH2::BVH2() : nodesAlloc(false)
{
}

void BVH2::build(std::vector<Triangle>& triangles)
{
	Log& log = App::getLog();

	Timer timer;
	uint32_t triangleCount = uint32_t(triangles.size());

	if (triangleCount == 0)
	{
		log.logWarning("Could not build BVH from empty triangle list");
		return;
	}

	log.logInfo("BVH2 building started (triangles: %d)", triangleCount);

	std::vector<BVHBuildTriangle> buildTriangles(triangleCount);
	std::vector<BVHSplitCache> cache(triangleCount);
	BVHSplitOutput splitOutput;

	for (uint32_t i = 0; i < triangleCount; ++i)
	{
		AABB aabb = triangles[i].getAABB();

		buildTriangles[i].triangle = &triangles[i];
		buildTriangles[i].aabb = aabb;
		buildTriangles[i].center = aabb.getCenter();
	}

	std::vector<BVHNode> nodes;
	nodes.reserve(triangleCount);

	BVH2BuildEntry stack[128];
	uint32_t stackIndex = 0;
	uint32_t nodeCount = 0;
	uint32_t leafCount = 0;

	// push to stack
	stack[stackIndex].start = 0;
	stack[stackIndex].end = triangleCount;
	stack[stackIndex].parent = -1;
	stackIndex++;

	while (stackIndex > 0)
	{
		nodeCount++;

		// pop from stack
		BVH2BuildEntry buildEntry = stack[--stackIndex];

		BVHNode node;
		node.rightOffset = -3;
		node.triangleOffset = uint32_t(buildEntry.start);
		node.triangleCount = uint32_t(buildEntry.end - buildEntry.start);
		node.splitAxis = 0;

		// leaf node
		if (node.triangleCount <= maxLeafSize)
			node.rightOffset = 0;

		// update the parent rightOffset when visiting its right child
		if (buildEntry.parent != -1)
		{
			uint32_t parent = uint32_t(buildEntry.parent);

			if (++nodes[parent].rightOffset == -1)
				nodes[parent].rightOffset = int32_t(nodeCount - 1 - parent);
		}

		if (node.rightOffset != 0)
		{
			splitOutput = BVH::calculateSplit(buildTriangles, cache, buildEntry.start, buildEntry.end);

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
		stack[stackIndex].parent = int32_t(nodeCount) - 1;
		stackIndex++;

		// push left child
		stack[stackIndex].start = buildEntry.start;
		stack[stackIndex].end = splitOutput.index;
		stack[stackIndex].parent = int32_t(nodeCount) - 1;
		stackIndex++;
	}

	if (nodes.size() > 0)
	{
		nodesAlloc.resize(nodes.size());
		nodesAlloc.write(nodes.data(), nodes.size());
	}

	std::vector<Triangle> sortedTriangles(triangleCount);

	for (uint32_t i = 0; i < triangleCount; ++i)
		sortedTriangles[i] = *buildTriangles[i].triangle;

	triangles = sortedTriangles;

	log.logInfo("BVH2 building finished (time: %s, nodes: %d, leafs: %d, triangles/leaf: %.2f)", timer.getElapsed().getString(true), nodeCount - leafCount, leafCount, float(triangleCount) / float(leafCount));
}

CUDA_CALLABLE bool BVH2::intersect(const Scene& scene, const Ray& ray, Intersection& intersection) const
{
	if (nodesAlloc.getPtr() == nullptr || scene.getTriangles() == nullptr)
		return false;

	if (ray.isVisibilityRay && intersection.wasFound)
		return true;

	uint32_t stack[64];
	uint32_t stackIndex = 0;
	bool wasFound = false;

	stack[stackIndex++] = 0;

	while (stackIndex > 0)
	{
		uint32_t nodeIndex = stack[--stackIndex];
		const BVHNode& node = nodesAlloc.getPtr()[nodeIndex];

		// leaf node
		if (node.rightOffset == 0)
		{
			for (uint32_t i = 0; i < node.triangleCount; ++i)
			{
				if (scene.getTriangles()[node.triangleOffset + i].intersect(scene, ray, intersection))
				{
					if (ray.isVisibilityRay)
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
				stack[stackIndex++] = nodeIndex + uint32_t(node.rightOffset); // right child
			}
			else
			{
				stack[stackIndex++] = nodeIndex + uint32_t(node.rightOffset); // right child
				stack[stackIndex++] = nodeIndex + 1; // left child
			}
		}
	}

	return wasFound;
}
