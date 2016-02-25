// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "BVH/BVH4.h"
#include "App.h"
#include "Utils/Log.h"
#include "Utils/Timer.h"
#include "Tracing/Scene.h"
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

void BVH4::build(std::vector<Triangle>& triangles, uint64_t maxLeafSize)
{
	Log& log = App::getLog();

	log.logInfo("BVH4 building started (triangles: %d)", triangles.size());

	Timer timer;
	uint64_t triangleCount = triangles.size();
	BVHBuildTriangleVector buildTriangles(triangleCount);
	std::vector<BVHSplitCache> cache(triangleCount);
	BVHSplitOutput splitOutputs[3];

	for (uint64_t i = 0; i < triangleCount; ++i)
	{
		Aabb aabb = triangles[i].getAabb();

		buildTriangles[i].triangle = &triangles[i];
		buildTriangles[i].aabb = aabb;
		buildTriangles[i].center = aabb.getCenter();
	}

	nodes.clear();
	nodes.reserve(triangleCount);

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
		nodeCount++;

		// pop from stack
		BVH4BuildEntry buildEntry = stack[--stackIndex];

		BVH4Node node;
		node.triangleOffset = uint32_t(buildEntry.start);
		node.triangleCount = uint32_t(buildEntry.end - buildEntry.start);
		node.isLeaf = (node.triangleCount <= maxLeafSize);

		if (buildEntry.parent != -1 && buildEntry.child != -1)
		{
			uint64_t parent = uint64_t(buildEntry.parent);
			uint64_t child = uint64_t(buildEntry.child);

			nodes[parent].rightOffset[child] = uint32_t(nodeCount - 1 - parent);
		}

		if (!node.isLeaf)
		{
			// middle split
			splitOutputs[1] = calculateSplit(buildTriangles, cache, buildEntry.start, buildEntry.end);

			// left split
			splitOutputs[0] = calculateSplit(buildTriangles, cache, buildEntry.start, splitOutputs[1].index);

			// right split
			splitOutputs[2] = calculateSplit(buildTriangles, cache, splitOutputs[1].index, buildEntry.end);

			node.aabbMinX[0] = splitOutputs[0].leftAabb.min.x;
			node.aabbMinY[0] = splitOutputs[0].leftAabb.min.y;
			node.aabbMinZ[0] = splitOutputs[0].leftAabb.min.z;
			node.aabbMaxX[0] = splitOutputs[0].leftAabb.max.x;
			node.aabbMaxY[0] = splitOutputs[0].leftAabb.max.y;
			node.aabbMaxZ[0] = splitOutputs[0].leftAabb.max.z;

			node.aabbMinX[1] = splitOutputs[0].rightAabb.min.x;
			node.aabbMinY[1] = splitOutputs[0].rightAabb.min.y;
			node.aabbMinZ[1] = splitOutputs[0].rightAabb.min.z;
			node.aabbMaxX[1] = splitOutputs[0].rightAabb.max.x;
			node.aabbMaxY[1] = splitOutputs[0].rightAabb.max.y;
			node.aabbMaxZ[1] = splitOutputs[0].rightAabb.max.z;

			node.aabbMinX[2] = splitOutputs[2].leftAabb.min.x;
			node.aabbMinY[2] = splitOutputs[2].leftAabb.min.y;
			node.aabbMinZ[2] = splitOutputs[2].leftAabb.min.z;
			node.aabbMaxX[2] = splitOutputs[2].leftAabb.max.x;
			node.aabbMaxY[2] = splitOutputs[2].leftAabb.max.y;
			node.aabbMaxZ[2] = splitOutputs[2].leftAabb.max.z;

			node.aabbMinX[3] = splitOutputs[2].rightAabb.min.x;
			node.aabbMinY[3] = splitOutputs[2].rightAabb.min.y;
			node.aabbMinZ[3] = splitOutputs[2].rightAabb.min.z;
			node.aabbMaxX[3] = splitOutputs[2].rightAabb.max.x;
			node.aabbMaxY[3] = splitOutputs[2].rightAabb.max.y;
			node.aabbMaxZ[3] = splitOutputs[2].rightAabb.max.z;
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

	nodes.shrink_to_fit();

	std::vector<Triangle> sortedTriangles(triangleCount);

	for (uint64_t i = 0; i < triangleCount; ++i)
		sortedTriangles[i] = *buildTriangles[i].triangle;

	triangles = sortedTriangles;

	log.logInfo("BVH4 building finished (time: %s, nodes: %d, leafs: %d, triangles/leaf: %.2f)", timer.getElapsed().getString(true), nodeCount - leafCount, leafCount, float(triangleCount) / float(leafCount));
}

bool BVH4::intersect(const Scene& scene, const Ray& ray, Intersection& intersection) const
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
		const BVH4Node& node = nodes[nodeIndex];

		if (node.isLeaf)
		{
			for (uint64_t i = 0; i < node.triangleCount; ++i)
			{
				if (scene.bvhData.triangles[node.triangleOffset + i].intersect(scene, ray, intersection))
				{
					if (ray.fastOcclusion)
						return true;

					wasFound = true;
				}
			}

			continue;
		}

		std::array<uint32_t, 4> intersects = Aabb::intersects(
			&node.aabbMinX[0],
			&node.aabbMinY[0],
			&node.aabbMinZ[0],
			&node.aabbMaxX[0],
			&node.aabbMaxY[0],
			&node.aabbMaxZ[0],
			ray);

		if (intersects[3])
			stack[stackIndex++] = nodeIndex + node.rightOffset[2];

		if (intersects[2])
			stack[stackIndex++] = nodeIndex + node.rightOffset[1];

		if (intersects[1])
			stack[stackIndex++] = nodeIndex + node.rightOffset[0];

		if (intersects[0])
			stack[stackIndex++] = nodeIndex + 1;
	}

	return wasFound;
}
