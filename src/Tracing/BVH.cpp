// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "stdafx.h"

#include "Tracing/BVH.h"
#include "Tracing/Scene.h"
#include "Tracing/Ray.h"
#include "Tracing/Intersection.h"
#include "Tracing/Triangle.h"
#include "App.h"
#include "Utils/Log.h"
#include "Math/Vector3.h"

using namespace Raycer;

bool BVH::intersect(const Ray& ray, Intersection& intersection) const
{
	if (ray.fastOcclusion && intersection.wasFound)
		return true;

	uint64_t stack[128];
	uint64_t stackptr = 0;
	bool wasFound = false;

	// push to stack
	stack[stackptr] = 0;
	stackptr++;

	while (stackptr > 0)
	{
		// pop from stack
		stackptr--;
		uint64_t index = stack[stackptr];
		BVHNode node = nodes[index];

		// leaf node -> intersect with all its primitives
		if (node.rightOffset == 0)
		{
			for (uint64_t i = 0; i < node.primitiveCount; ++i)
			{
				if (orderedTriangles[node.startOffset + i]->intersect(ray, intersection))
				{
					if (ray.fastOcclusion)
						return true;

					wasFound = true;
				}
			}
		}
		else // travel down the tree
		{
			// right child
			if (nodes[index + uint64_t(node.rightOffset)].aabb.intersects(ray))
			{
				stack[stackptr] = index + uint64_t(node.rightOffset);
				stackptr++;
			}

			// left child
			if (nodes[index + 1].aabb.intersects(ray))
			{
				stack[stackptr] = index + 1;
				stackptr++;
			}
		}
	}

	return wasFound;
}

void BVH::build(const std::vector<Triangle>& triangles, const BVHBuildInfo& buildInfo)
{
	Log& log = App::getLog();

	log.logInfo("Building BVH (triangles: %d)", triangles.size());

	auto startTime = std::chrono::high_resolution_clock::now();

	std::random_device rd;
	std::mt19937 generator(rd());

	BVHBuildEntry stack[128];
	orderedTriangles.clear();
	nodes.clear();

	orderedTriangles.reserve(triangles.size());

	for (const Triangle& triangle : triangles)
		orderedTriangles.push_back(&triangle);

	uint64_t stackptr = 0;
	uint64_t nodeCount = 0;
	uint64_t leafCount = 0;
	uint64_t actualNodeCount = 0;

	enum { ROOT = -4, UNVISITED = -3, VISITED_TWICE = -1 };

	// push to stack
	stack[stackptr].start = 0;
	stack[stackptr].end = orderedTriangles.size();
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
		node.primitiveCount = buildEntry.end - buildEntry.start;

		for (uint64_t i = buildEntry.start; i < buildEntry.end; ++i)
			node.aabb.expand(orderedTriangles[i]->getAABB());

		// leaf node indicated by rightOffset == 0
		if (node.primitiveCount <= buildInfo.maxLeafSize)
		{
			node.rightOffset = 0;
			leafCount++;
		}

		nodes.push_back(node);

		// update the parent rightOffset when visiting its right child
		if (buildEntry.parent != ROOT)
		{
			nodes[uint64_t(buildEntry.parent)].rightOffset++;

			if (nodes[uint64_t(buildEntry.parent)].rightOffset == VISITED_TWICE)
				nodes[uint64_t(buildEntry.parent)].rightOffset = int64_t(nodeCount) - 1 - buildEntry.parent;
		}

		// leaf node -> no further subdivision
		if (node.rightOffset == 0)
			continue;

		uint64_t axis;
		double splitPoint;
		actualNodeCount++;

		if (buildInfo.useSAH)
			calculateSAHSplit(axis, splitPoint, node.aabb, buildInfo, buildEntry);
		else
			calculateSplit(axis, splitPoint, node.aabb, buildInfo, buildEntry, generator);

		uint64_t middle = buildEntry.start;

		// partition primitive range by the split point
		for (uint64_t i = buildEntry.start; i < buildEntry.end; ++i)
		{
			if (orderedTriangles[i]->getAABB().getCenter().get(axis) <= splitPoint)
			{
				std::swap(orderedTriangles[i], orderedTriangles[middle]);
				middle++;
			}
		}

		// partition failed -> fallback
		if (middle == buildEntry.start || middle == buildEntry.end)
			middle = buildEntry.start + (buildEntry.end - buildEntry.start) / 2;

		// push right child
		stack[stackptr].start = middle;
		stack[stackptr].end = buildEntry.end;
		stack[stackptr].parent = int64_t(nodeCount) - 1;
		stackptr++;

		// push left child
		stack[stackptr].start = buildEntry.start;
		stack[stackptr].end = middle;
		stack[stackptr].parent = int64_t(nodeCount) - 1;
		stackptr++;
	}

	hasBeenBuilt = true;

	orderedTriangleIds.clear();

	for (const Triangle* triangle : orderedTriangles)
		orderedTriangleIds.push_back(triangle->id);

	auto elapsedTime = std::chrono::high_resolution_clock::now() - startTime;
	auto milliseconds = std::chrono::duration_cast<std::chrono::milliseconds>(elapsedTime).count();

	log.logInfo("BVH building finished (time: %d ms, nodes: %d, leafs: %d)", milliseconds, nodeCount, leafCount);
}

void BVH::restore(const Scene& scene)
{
	orderedTriangles.clear();

	for (uint64_t triangleId : orderedTriangleIds)
		orderedTriangles.push_back(scene.trianglesMap.at(triangleId));

	return;
}

void BVH::calculateSplit(uint64_t& axis, double& splitPoint, const AABB& nodeAABB, const BVHBuildInfo& buildInfo, const BVHBuildEntry& buildEntry, std::mt19937& generator)
{
	if (buildInfo.axisSelection == BVHAxisSelection::LARGEST)
		axis = nodeAABB.getLargestAxis();
	else if (buildInfo.axisSelection == BVHAxisSelection::RANDOM)
	{
		std::uniform_int_distribution<uint64_t> randomAxis(0, 2);
		axis = randomAxis(generator);
	}
	else
		throw std::runtime_error("Unknown BVH axis selection");

	if (buildInfo.axisSplit == BVHAxisSplit::MIDDLE)
		splitPoint = nodeAABB.getCenter().get(axis);
	else if (buildInfo.axisSplit == BVHAxisSplit::MEDIAN)
		splitPoint = calculateMedianPoint(axis, buildEntry);
	else if (buildInfo.axisSplit == BVHAxisSplit::RANDOM)
	{
		std::uniform_real_distribution<double> randomDouble(nodeAABB.getMin().get(axis), nodeAABB.getMax().get(axis));
		splitPoint = randomDouble(generator);
	}
	else
		throw std::runtime_error("Unknown BVH axis split");
}

void BVH::calculateSAHSplit(uint64_t& axis, double& splitPoint, const AABB& nodeAABB, const BVHBuildInfo& buildInfo, const BVHBuildEntry& buildEntry)
{
	double lowestScore = std::numeric_limits<double>::max();

	for (uint64_t tempAxis = 0; tempAxis <= 2; ++tempAxis)
	{
		double tempSplitPoint = nodeAABB.getCenter().get(tempAxis);
		double score = calculateSAHScore(tempAxis, tempSplitPoint, nodeAABB, buildEntry);

		if (score < lowestScore)
		{
			axis = tempAxis;
			splitPoint = tempSplitPoint;
			lowestScore = score;
		}

		tempSplitPoint = calculateMedianPoint(tempAxis, buildEntry);
		score = calculateSAHScore(tempAxis, tempSplitPoint, nodeAABB, buildEntry);

		if (score < lowestScore)
		{
			axis = tempAxis;
			splitPoint = tempSplitPoint;
			lowestScore = score;
		}

		if (buildInfo.regularSAHSplits > 0)
		{
			double step = nodeAABB.getExtent().get(tempAxis) / double(buildInfo.regularSAHSplits);
			tempSplitPoint = nodeAABB.getMin().get(tempAxis);

			for (uint64_t i = 0; i < buildInfo.regularSAHSplits - 1; ++i)
			{
				tempSplitPoint += step;
				score = calculateSAHScore(tempAxis, tempSplitPoint, nodeAABB, buildEntry);

				if (score < lowestScore)
				{
					axis = tempAxis;
					splitPoint = tempSplitPoint;
					lowestScore = score;
				}
			}
		}
	}
}

double BVH::calculateSAHScore(uint64_t axis, double splitPoint, const AABB& nodeAABB, const BVHBuildEntry& buildEntry)
{
	assert(buildEntry.end != buildEntry.start);

	AABB leftAABB, rightAABB;
	uint64_t leftCount = 0;
	uint64_t rightCount = 0;

	for (uint64_t i = buildEntry.start; i < buildEntry.end; ++i)
	{
		AABB primitiveAABB = orderedTriangles[i]->getAABB();

		if (primitiveAABB.getCenter().get(axis) <= splitPoint)
		{
			leftAABB.expand(primitiveAABB);
			leftCount++;
		}
		else
		{
			rightAABB.expand(primitiveAABB);
			rightCount++;
		}
	}

	double score = 0.0;

	if (leftCount > 0)
		score += (leftAABB.getSurfaceArea() / nodeAABB.getSurfaceArea()) * double(leftCount);

	if (rightCount > 0)
		score += (rightAABB.getSurfaceArea() / nodeAABB.getSurfaceArea()) * double(rightCount);

	return score;
}

double BVH::calculateMedianPoint(uint64_t axis, const BVHBuildEntry& buildEntry)
{
	std::vector<double> centerPoints;

	for (uint64_t i = buildEntry.start; i < buildEntry.end; ++i)
		centerPoints.push_back(orderedTriangles[i]->getAABB().getCenter().get(axis));

	std::sort(centerPoints.begin(), centerPoints.end());
	uint64_t size = centerPoints.size();
	double median;

	assert(size >= 2);

	if (size % 2 == 0)
		median = (centerPoints[size / 2 - 1] + centerPoints[size / 2]) / 2.0;
	else
		median = centerPoints[size / 2];

	return median;
}
