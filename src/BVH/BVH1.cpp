// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "BVH/BVH1.h"
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
	struct BVH1BuildEntry
	{
		uint64_t start;
		uint64_t end;
		int64_t parent;
	};
}

void BVH1::build(Scene& scene)
{
	Log& log = App::getLog();

	Timer timer;
	uint64_t triangleCount = scene.bvhData.triangles.size();

	log.logInfo("BVH1 building started (triangles: %d)", triangleCount);

	std::vector<BVHBuildTriangle> buildTriangles(triangleCount);
	std::vector<BVHSplitCache> cache(triangleCount);
	BVHSplitOutput splitOutput;

	for (uint64_t i = 0; i < triangleCount; ++i)
	{
		AABB aabb = scene.bvhData.triangles[i].getAABB();

		buildTriangles[i].triangle = &scene.bvhData.triangles[i];
		buildTriangles[i].aabb = aabb;
		buildTriangles[i].center = aabb.getCenter();
	}

	nodes.clear();
	nodes.reserve(triangleCount);

	scene.bvhData.triangles4.clear();
	scene.bvhData.triangles4.reserve(triangleCount / 4);

	BVH1BuildEntry stack[128];
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
		BVH1BuildEntry buildEntry = stack[--stackIndex];

		BVHNode node;
		node.rightOffset = -3;
		node.triangleOffset = uint32_t(buildEntry.start);
		node.triangleCount = uint32_t(buildEntry.end - buildEntry.start);
		node.splitAxis = 0;

		if (node.triangleCount <= 8)
		{
			node.rightOffset = 0;

			TriangleSOA<8> triangleSOA;
			memset(&triangleSOA, 0, sizeof(TriangleSOA<8>));

			uint64_t index = 0;

			for (uint64_t i = buildEntry.start; i < buildEntry.end; ++i)
			{
				Triangle triangle = *buildTriangles[i].triangle;

				triangleSOA.vertex1X[index] = triangle.vertices[0].x;
				triangleSOA.vertex1Y[index] = triangle.vertices[0].y;
				triangleSOA.vertex1Z[index] = triangle.vertices[0].z;
				triangleSOA.vertex2X[index] = triangle.vertices[1].x;
				triangleSOA.vertex2Y[index] = triangle.vertices[1].y;
				triangleSOA.vertex2Z[index] = triangle.vertices[1].z;
				triangleSOA.vertex3X[index] = triangle.vertices[2].x;
				triangleSOA.vertex3Y[index] = triangle.vertices[2].y;
				triangleSOA.vertex3Z[index] = triangle.vertices[2].z;
				triangleSOA.triangleId[index] = uint32_t(triangle.id);

				index++;
			}

			node.triangleOffset = uint32_t(scene.bvhData.triangles8.size());
			scene.bvhData.triangles8.push_back(triangleSOA);
		}

		// update the parent rightOffset when visiting its right child
		if (buildEntry.parent != -1)
		{
			uint64_t parent = uint64_t(buildEntry.parent);

			if (++nodes[parent].rightOffset == -1)
				nodes[parent].rightOffset = int32_t(nodeCount - 1 - parent);
		}

		if (node.rightOffset != 0)
		{
			splitOutput = calculateSplit(buildTriangles, cache, buildEntry.start, buildEntry.end);

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

	std::vector<Triangle> sortedTriangles(triangleCount);
	
	for (uint64_t i = 0; i < triangleCount; ++i)
		sortedTriangles[i] = *buildTriangles[i].triangle;

	scene.bvhData.triangles = sortedTriangles;

	log.logInfo("BVH1 building finished (time: %s, nodes: %d, leafs: %d, triangles/leaf: %.2f)", timer.getElapsed().getString(true), nodeCount - leafCount, leafCount, float(triangleCount) / float(leafCount));
}

bool BVH1::intersect(const Scene& scene, const Ray& ray, Intersection& intersection) const
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
		const BVHNode& node = nodes[nodeIndex];

		// leaf node
		if (node.rightOffset == 0)
		{
			/*for (uint64_t i = 0; i < node.triangleCount; ++i)
			{
				if (scene.bvhData.triangles[node.triangleOffset + i].intersect(scene, ray, intersection))
				{
					if (ray.fastOcclusion)
						return true;

					wasFound = true;
				}
			}*/

			const TriangleSOA<8>& triangleSOA = scene.bvhData.triangles8[node.triangleOffset];

			if (Triangle::intersect(
				&triangleSOA.vertex1X[0],
				&triangleSOA.vertex1Y[0],
				&triangleSOA.vertex1Z[0],
				&triangleSOA.vertex2X[0],
				&triangleSOA.vertex2Y[0],
				&triangleSOA.vertex2Z[0],
				&triangleSOA.vertex3X[0],
				&triangleSOA.vertex3Y[0],
				&triangleSOA.vertex3Z[0],
				&triangleSOA.triangleId[0],
				scene,
				ray,
				intersection))
			{
				if (ray.fastOcclusion)
					return true;

				wasFound = true;
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
