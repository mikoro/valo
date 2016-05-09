// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "BVH/BVH8.h"
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
	struct BVH8BuildEntry
	{
		uint32_t start;
		uint32_t end;
		int32_t parent;
		int32_t child;
	};
}

BVH8::BVH8() : nodesAlloc(false), triangles8Alloc(false)
{
}

void BVH8::build(std::vector<Triangle>& triangles)
{
	Log& log = App::getLog();

	Timer timer;
	uint32_t triangleCount = uint32_t(triangles.size());

	if (triangleCount == 0)
	{
		log.logWarning("Could not build BVH from empty triangle list");
		return;
	}

	log.logInfo("BVH8 building started (triangles: %d)", triangleCount);

	std::vector<BVHBuildTriangle> buildTriangles(triangleCount);
	std::vector<BVHSplitCache> cache(triangleCount);
	BVHSplitOutput splitOutputs[7];

	// build triangles only contain necessary data (will be faster to sort)
	for (uint32_t i = 0; i < triangleCount; ++i)
	{
		AABB aabb = triangles[i].getAABB();

		buildTriangles[i].triangle = &triangles[i];
		buildTriangles[i].aabb = aabb;
		buildTriangles[i].center = aabb.getCenter();
	}

	std::vector<BVHNodeSOA<8>> nodes;
	std::vector<TriangleSOA<8>> triangles8;

	nodes.reserve(triangleCount);
	triangles8.reserve(triangleCount / 8);

	BVH8BuildEntry stack[128];
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
		BVH8BuildEntry buildEntry = stack[--stackIndex];

		BVHNodeSOA<8> node;
		node.triangleOffset = 0;
		node.triangleCount = uint32_t(buildEntry.end - buildEntry.start);
		node.isLeaf = node.triangleCount <= 8;

		// if not the leftmost child, adjust the according offset at the parent
		if (buildEntry.parent != -1 && buildEntry.child != -1)
		{
			uint32_t parent = uint32_t(buildEntry.parent);
			uint32_t child = uint32_t(buildEntry.child);

			nodes[parent].rightOffset[child] = uint32_t(nodeCount - 1 - parent);
		}

		uint32_t splitIndex[9] = { 0 };

		auto calculateAABB = [&](uint32_t start, uint32_t end)
		{
			AABB aabb;

			for (uint32_t i = start; i < end; ++i)
				aabb.expand(buildTriangles[i].aabb);

			return aabb;
		};

		auto setAABB = [&node](uint32_t aabbIndex, const AABB& aabb)
		{
			node.aabbMinX[aabbIndex] = aabb.min.x;
			node.aabbMinY[aabbIndex] = aabb.min.y;
			node.aabbMinZ[aabbIndex] = aabb.min.z;
			node.aabbMaxX[aabbIndex] = aabb.max.x;
			node.aabbMaxY[aabbIndex] = aabb.max.y;
			node.aabbMaxZ[aabbIndex] = aabb.max.z;
		};

		if (node.isLeaf)
		{
			TriangleSOA<8> triangleSOA;
			memset(&triangleSOA, 0, sizeof(TriangleSOA<8>));

			for (uint32_t i = buildEntry.start, j = 0; i < buildEntry.end; ++i, ++j)
			{
				Triangle& triangle = *buildTriangles[i].triangle;

				triangleSOA.vertex1X[j] = triangle.vertices[0].x;
				triangleSOA.vertex1Y[j] = triangle.vertices[0].y;
				triangleSOA.vertex1Z[j] = triangle.vertices[0].z;
				triangleSOA.vertex2X[j] = triangle.vertices[1].x;
				triangleSOA.vertex2Y[j] = triangle.vertices[1].y;
				triangleSOA.vertex2Z[j] = triangle.vertices[1].z;
				triangleSOA.vertex3X[j] = triangle.vertices[2].x;
				triangleSOA.vertex3Y[j] = triangle.vertices[2].y;
				triangleSOA.vertex3Z[j] = triangle.vertices[2].z;
				triangleSOA.triangleIndex[j] = uint32_t(i);
			}

			if (node.triangleCount > 0)
			{
				node.triangleOffset = uint32_t(triangles8.size());
				triangles8.push_back(triangleSOA);
			}
		}
		else if (node.triangleCount <= 64) // static split
		{
			splitIndex[0] = buildEntry.start;
			splitIndex[1] = MIN(buildEntry.start + 8, buildEntry.end);
			splitIndex[2] = MIN(buildEntry.start + 16, buildEntry.end);
			splitIndex[3] = MIN(buildEntry.start + 24, buildEntry.end);
			splitIndex[4] = MIN(buildEntry.start + 32, buildEntry.end);
			splitIndex[5] = MIN(buildEntry.start + 40, buildEntry.end);
			splitIndex[6] = MIN(buildEntry.start + 48, buildEntry.end);
			splitIndex[7] = MIN(buildEntry.start + 56, buildEntry.end);
			splitIndex[8] = buildEntry.end;

			setAABB(0, calculateAABB(splitIndex[0], splitIndex[1]));
			setAABB(1, calculateAABB(splitIndex[1], splitIndex[2]));
			setAABB(2, calculateAABB(splitIndex[2], splitIndex[3]));
			setAABB(3, calculateAABB(splitIndex[3], splitIndex[4]));
			setAABB(4, calculateAABB(splitIndex[4], splitIndex[5]));
			setAABB(5, calculateAABB(splitIndex[5], splitIndex[6]));
			setAABB(6, calculateAABB(splitIndex[6], splitIndex[7]));
			setAABB(7, calculateAABB(splitIndex[7], splitIndex[8]));
		}
		else // SAH split
		{
			splitOutputs[3] = BVH::calculateSplit(buildTriangles, cache, buildEntry.start, buildEntry.end);
			splitOutputs[1] = BVH::calculateSplit(buildTriangles, cache, buildEntry.start, splitOutputs[3].index);
			splitOutputs[0] = BVH::calculateSplit(buildTriangles, cache, buildEntry.start, splitOutputs[1].index);
			splitOutputs[2] = BVH::calculateSplit(buildTriangles, cache, splitOutputs[1].index, splitOutputs[3].index);
			splitOutputs[5] = BVH::calculateSplit(buildTriangles, cache, splitOutputs[3].index, buildEntry.end);
			splitOutputs[6] = BVH::calculateSplit(buildTriangles, cache, splitOutputs[5].index, buildEntry.end);
			splitOutputs[4] = BVH::calculateSplit(buildTriangles, cache, splitOutputs[3].index, splitOutputs[5].index);

			splitIndex[0] = buildEntry.start;
			splitIndex[1] = splitOutputs[0].index;
			splitIndex[2] = splitOutputs[1].index;
			splitIndex[3] = splitOutputs[2].index;
			splitIndex[4] = splitOutputs[3].index;
			splitIndex[5] = splitOutputs[4].index;
			splitIndex[6] = splitOutputs[5].index;
			splitIndex[7] = splitOutputs[6].index;
			splitIndex[8] = buildEntry.end;

			node.splitAxis[0] = uint16_t(splitOutputs[0].axis);
			node.splitAxis[1] = uint16_t(splitOutputs[1].axis);
			node.splitAxis[2] = uint16_t(splitOutputs[2].axis);
			node.splitAxis[3] = uint16_t(splitOutputs[3].axis);
			node.splitAxis[4] = uint16_t(splitOutputs[4].axis);
			node.splitAxis[5] = uint16_t(splitOutputs[5].axis);
			node.splitAxis[6] = uint16_t(splitOutputs[6].axis);

			setAABB(0, splitOutputs[0].leftAABB);
			setAABB(1, splitOutputs[0].rightAABB);
			setAABB(2, splitOutputs[2].leftAABB);
			setAABB(3, splitOutputs[2].rightAABB);
			setAABB(4, splitOutputs[4].leftAABB);
			setAABB(5, splitOutputs[4].rightAABB);
			setAABB(6, splitOutputs[6].leftAABB);
			setAABB(7, splitOutputs[6].rightAABB);
		}

		nodes.push_back(node);

		if (node.isLeaf)
		{
			leafCount++;
			continue;
		}

		auto pushChild = [&](uint32_t start, uint32_t end, int32_t child)
		{
			stack[stackIndex].start = start;
			stack[stackIndex].end = end;
			stack[stackIndex].parent = int32_t(nodeCount) - 1;
			stack[stackIndex].child = child;
			stackIndex++;
		};

		pushChild(splitIndex[7], splitIndex[8], 6);
		pushChild(splitIndex[6], splitIndex[7], 5);
		pushChild(splitIndex[5], splitIndex[6], 4);
		pushChild(splitIndex[4], splitIndex[5], 3);
		pushChild(splitIndex[3], splitIndex[4], 2);
		pushChild(splitIndex[2], splitIndex[3], 1);
		pushChild(splitIndex[1], splitIndex[2], 0);
		pushChild(splitIndex[0], splitIndex[1], -1);
	}

	if (nodes.size() > 0)
	{
		nodesAlloc.resize(nodes.size());
		nodesAlloc.write(nodes.data(), nodes.size());
	}

	if (triangles8.size() > 0)
	{
		triangles8Alloc.resize(triangles8.size());
		triangles8Alloc.write(triangles8.data(), triangles8.size());
	}

	std::vector<Triangle> sortedTriangles(triangleCount);

	for (uint32_t i = 0; i < triangleCount; ++i)
		sortedTriangles[i] = *buildTriangles[i].triangle;

	triangles = sortedTriangles;

	log.logInfo("BVH8 building finished (time: %s, nodes: %d, leafs: %d, triangles/leaf: %.2f)", timer.getElapsed().getString(true), nodeCount - leafCount, leafCount, float(triangleCount) / float(leafCount));
}

CUDA_CALLABLE bool BVH8::intersect(const Scene& scene, const Ray& ray, Intersection& intersection) const
{
	if (nodesAlloc.getPtr() == nullptr || triangles8Alloc.getPtr() == nullptr || scene.getTriangles() == nullptr)
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
		const BVHNodeSOA<8>& node = nodesAlloc.getPtr()[nodeIndex];

		if (node.isLeaf)
		{
			if (node.triangleCount == 0)
				continue;

			const TriangleSOA<8>& triangle = triangles8Alloc.getPtr()[node.triangleOffset];

			if (Triangle::intersect<8>(
				triangle.vertex1X,
				triangle.vertex1Y,
				triangle.vertex1Z,
				triangle.vertex2X,
				triangle.vertex2Y,
				triangle.vertex2Z,
				triangle.vertex3X,
				triangle.vertex3Y,
				triangle.vertex3Z,
				triangle.triangleIndex,
				scene,
				ray,
				intersection))
			{
				if (ray.isVisibilityRay)
					return true;

				wasFound = true;
			}

			continue;
		}

		ALIGN(16) bool intersects[8];

		AABB::intersects<8>(
			node.aabbMinX,
			node.aabbMinY,
			node.aabbMinZ,
			node.aabbMaxX,
			node.aabbMaxY,
			node.aabbMaxZ,
			intersects,
			ray);

		if (intersects[7])
			stack[stackIndex++] = nodeIndex + node.rightOffset[6];

		if (intersects[6])
			stack[stackIndex++] = nodeIndex + node.rightOffset[5];

		if (intersects[5])
			stack[stackIndex++] = nodeIndex + node.rightOffset[4];

		if (intersects[4])
			stack[stackIndex++] = nodeIndex + node.rightOffset[3];

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
