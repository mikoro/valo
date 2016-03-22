// Copyright © 2016 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Core/Precompiled.h"

#include "BVH/BVH4.h"
#include "Core/App.h"
#include "Core/Common.h"
#include "Utils/Log.h"
#include "Utils/Timer.h"
#include "Core/Scene.h"
#include "Core/Triangle.h"
#include "Core/Ray.h"
#include "Core/Intersection.h"

using namespace Raycer;

namespace
{
	struct BVH4BuildEntry
	{
		uint32_t start;
		uint32_t end;
		int32_t parent;
		int32_t child;
	};
}

BVH4::~BVH4()
{
	RAYCER_FREE(nodesPtr);
	RAYCER_FREE(triangles4Ptr);
}

void BVH4::build(std::vector<Triangle>& triangles)
{
	Log& log = App::getLog();

	Timer timer;
	uint32_t triangleCount = uint32_t(triangles.size());

	if (triangleCount == 0)
	{
		log.logWarning("Could not build BVH from empty triangle list");
		return;
	}

	log.logInfo("BVH4 building started (triangles: %d)", triangleCount);

	std::vector<BVHBuildTriangle> buildTriangles(triangleCount);
	std::vector<BVHSplitCache> cache(triangleCount);
	BVHSplitOutput splitOutputs[3];

	// build triangles only contain necessary data (will be faster to sort)
	for (uint32_t i = 0; i < triangleCount; ++i)
	{
		AABB aabb = triangles[i].getAABB();

		buildTriangles[i].triangle = &triangles[i];
		buildTriangles[i].aabb = aabb;
		buildTriangles[i].center = aabb.getCenter();
	}

	std::vector<BVHNodeSOA<4>> nodes;
	std::vector<TriangleSOA<4>> triangles4;

	nodes.reserve(triangleCount);
	triangles4.reserve(triangleCount / 4);

	BVH4BuildEntry stack[128];
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
		BVH4BuildEntry buildEntry = stack[--stackIndex];

		BVHNodeSOA<4> node;
		node.triangleOffset = 0;
		node.triangleCount = uint32_t(buildEntry.end - buildEntry.start);
		node.isLeaf = node.triangleCount <= 4;

		// if not the leftmost child, adjust the according offset at the parent
		if (buildEntry.parent != -1 && buildEntry.child != -1)
		{
			uint32_t parent = uint32_t(buildEntry.parent);
			uint32_t child = uint32_t(buildEntry.child);

			nodes[parent].rightOffset[child] = uint32_t(nodeCount - 1 - parent);
		}

		uint32_t splitIndex1 = 0;
		uint32_t splitIndex2 = 0;
		uint32_t splitIndex3 = 0;
		uint32_t splitIndex4 = 0;
		uint32_t splitIndex5 = 0;

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
			TriangleSOA<4> triangleSOA;
			memset(&triangleSOA, 0, sizeof(TriangleSOA<4>));

			uint32_t index = 0;

			for (uint32_t i = buildEntry.start; i < buildEntry.end; ++i)
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
				triangleSOA.triangleIndex[index] = uint32_t(i);

				index++;
			}

			if (node.triangleCount > 0)
			{
				node.triangleOffset = uint32_t(triangles4.size());
				triangles4.push_back(triangleSOA);
			}
		}
		else if (node.triangleCount <= 16) // try to prevent leafs with small sizes
		{
			splitIndex1 = buildEntry.start;
			splitIndex2 = MIN(buildEntry.start + 4, buildEntry.end);
			splitIndex3 = MIN(buildEntry.start + 8, buildEntry.end);
			splitIndex4 = MIN(buildEntry.start + 12, buildEntry.end);
			splitIndex5 = buildEntry.end;

			setAABB(0, calculateAABB(splitIndex1, splitIndex2));
			setAABB(1, calculateAABB(splitIndex2, splitIndex3));
			setAABB(2, calculateAABB(splitIndex3, splitIndex4));
			setAABB(3, calculateAABB(splitIndex4, splitIndex5));
		}
		else // split into four parts using SAH
		{
			// middle split
			splitOutputs[1] = BVH::calculateSplit(buildTriangles, cache, buildEntry.start, buildEntry.end);

			// left split
			splitOutputs[0] = BVH::calculateSplit(buildTriangles, cache, buildEntry.start, splitOutputs[1].index);

			// right split
			splitOutputs[2] = BVH::calculateSplit(buildTriangles, cache, splitOutputs[1].index, buildEntry.end);

			// not used atm
			node.splitAxis[0] = uint16_t(splitOutputs[0].axis);
			node.splitAxis[1] = uint16_t(splitOutputs[1].axis);
			node.splitAxis[2] = uint16_t(splitOutputs[2].axis);

			setAABB(0, splitOutputs[0].leftAABB);
			setAABB(1, splitOutputs[0].rightAABB);
			setAABB(2, splitOutputs[2].leftAABB);
			setAABB(3, splitOutputs[2].rightAABB);

			splitIndex1 = buildEntry.start;
			splitIndex2 = splitOutputs[0].index;
			splitIndex3 = splitOutputs[1].index;
			splitIndex4 = splitOutputs[2].index;
			splitIndex5 = buildEntry.end;
		}

		nodes.push_back(node);

		if (node.isLeaf)
		{
			leafCount++;
			continue;
		}

		// push right child 2
		stack[stackIndex].start = splitIndex4;
		stack[stackIndex].end = splitIndex5;
		stack[stackIndex].parent = int32_t(nodeCount) - 1;
		stack[stackIndex].child = 2;
		stackIndex++;

		// push right child 1
		stack[stackIndex].start = splitIndex3;
		stack[stackIndex].end = splitIndex4;
		stack[stackIndex].parent = int32_t(nodeCount) - 1;
		stack[stackIndex].child = 1;
		stackIndex++;

		// push right child 0
		stack[stackIndex].start = splitIndex2;
		stack[stackIndex].end = splitIndex3;
		stack[stackIndex].parent = int32_t(nodeCount) - 1;
		stack[stackIndex].child = 0;
		stackIndex++;

		// push left child
		stack[stackIndex].start = splitIndex1;
		stack[stackIndex].end = splitIndex2;
		stack[stackIndex].parent = int32_t(nodeCount) - 1;
		stack[stackIndex].child = -1;
		stackIndex++;
	}

	nodesPtr = static_cast<BVHNodeSOA<4>*>(RAYCER_MALLOC(nodes.size() * sizeof(BVHNodeSOA<4>)));

	if (nodesPtr == nullptr)
		throw std::runtime_error("Could not allocate memory for BVH nodes");

	memcpy(nodesPtr, nodes.data(), nodes.size() * sizeof(BVHNodeSOA<4>));

	triangles4Ptr = static_cast<TriangleSOA<4>*>(RAYCER_MALLOC(triangles4.size() * sizeof(TriangleSOA<4>)));

	if (triangles4Ptr == nullptr)
		throw std::runtime_error("Could not allocate memory for BVH triangles");

	memcpy(triangles4Ptr, triangles4.data(), triangles4.size() * sizeof(TriangleSOA<4>));

	std::vector<Triangle> sortedTriangles(triangleCount);

	for (uint32_t i = 0; i < triangleCount; ++i)
		sortedTriangles[i] = *buildTriangles[i].triangle;

	triangles = sortedTriangles;

	log.logInfo("BVH4 building finished (time: %s, nodes: %d, leafs: %d, triangles/leaf: %.2f)", timer.getElapsed().getString(true), nodeCount - leafCount, leafCount, float(triangleCount) / float(leafCount));
}

CUDA_CALLABLE bool BVH4::intersect(const Scene& scene, const Ray& ray, Intersection& intersection) const
{
	if (nodesPtr == nullptr || triangles4Ptr == nullptr)
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
		const BVHNodeSOA<4>& node = nodesPtr[nodeIndex];

		if (node.isLeaf)
		{
			if (node.triangleCount == 0)
				continue;

			const TriangleSOA<4>& triangleSOA = triangles4Ptr[node.triangleOffset];

			if (Triangle::intersect<4>(
				triangleSOA.vertex1X,
				triangleSOA.vertex1Y,
				triangleSOA.vertex1Z,
				triangleSOA.vertex2X,
				triangleSOA.vertex2Y,
				triangleSOA.vertex2Z,
				triangleSOA.vertex3X,
				triangleSOA.vertex3Y,
				triangleSOA.vertex3Z,
				triangleSOA.triangleIndex,
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

		ALIGN(16) bool intersects[4];

		AABB::intersects<4>(
			node.aabbMinX,
			node.aabbMinY,
			node.aabbMinZ,
			node.aabbMaxX,
			node.aabbMaxY,
			node.aabbMaxZ,
			intersects,
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
