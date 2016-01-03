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
				for (uint64_t i = 0; i < node.primitiveCount; ++i)
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

					// left child
					stack[stackIndex] = nodeIndex + 1;
					stackIndex++;

					// right child
					stack[stackIndex] = nodeIndex + uint64_t(node.rightOffset);
					stackIndex++;
				}
				else
				{
					// right child
					stack[stackIndex] = nodeIndex + uint64_t(node.rightOffset);
					stackIndex++;

					// left child
					stack[stackIndex] = nodeIndex + 1;
					stackIndex++;
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
	uint64_t actualNodeCount = 0;

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
		node.primitiveCount = buildEntry.end - buildEntry.start;
		node.splitAxis = 0;

		for (uint64_t i = buildEntry.start; i < buildEntry.end; ++i)
			node.aabb.expand(triangles[i].getAABB());

		// leaf node indicated by rightOffset == 0
		if (node.primitiveCount <= buildInfo.maxLeafSize)
		{
			node.rightOffset = 0;
			leafCount++;
		}

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
			continue;
		}

		double splitPoint;
		actualNodeCount++;

		if (buildInfo.useSAH)
			calculateSAHSplit(triangles, node.splitAxis, splitPoint, node.aabb, buildInfo, buildEntry);
		else
			calculateSplit(triangles, node.splitAxis, splitPoint, node.aabb, buildInfo, buildEntry, random);

		nodes.push_back(node);

		uint64_t middle = buildEntry.start;

		// partition primitive range by the split point
		for (uint64_t i = buildEntry.start; i < buildEntry.end; ++i)
		{
			if (triangles[i].getAABB().getCenter().get(node.splitAxis) <= splitPoint)
			{
				std::swap(triangles[i], triangles[middle]);
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

	bvhHasBeenBuilt = true;

	log.logInfo("BVH building finished (time: %.2f ms, nodes: %d, leafs: %d)", timer.getElapsedMilliseconds(), nodeCount, leafCount);
}

bool BVH::hasBeenBuilt() const
{
	return bvhHasBeenBuilt;
}

void BVH::calculateSplit(const std::vector<Triangle>& triangles, uint64_t& splitAxis, double& splitPoint, const AABB& nodeAABB, const BVHBuildInfo& buildInfo, const BVHBuildEntry& buildEntry, Random& random)
{
	if (buildInfo.axisSelection == BVHAxisSelection::LARGEST)
		splitAxis = nodeAABB.getLargestAxis();
	else if (buildInfo.axisSelection == BVHAxisSelection::RANDOM)
		splitAxis = random.getUint64(0, 2);
	else
		throw std::runtime_error("Unknown BVH axis selection");

	if (buildInfo.axisSplit == BVHAxisSplit::MIDDLE)
		splitPoint = nodeAABB.getCenter().get(splitAxis);
	else if (buildInfo.axisSplit == BVHAxisSplit::MEDIAN)
		splitPoint = calculateMedianPoint(triangles, splitAxis, buildEntry);
	else if (buildInfo.axisSplit == BVHAxisSplit::RANDOM)
		splitPoint = random.getDouble(nodeAABB.getMin().get(splitAxis), nodeAABB.getMax().get(splitAxis));
	else
		throw std::runtime_error("Unknown BVH axis split");
}

void BVH::calculateSAHSplit(const std::vector<Triangle>& triangles, uint64_t& splitAxis, double& splitPoint, const AABB& nodeAABB, const BVHBuildInfo& buildInfo, const BVHBuildEntry& buildEntry)
{
	double lowestScore = std::numeric_limits<double>::max();

	for (uint64_t tempAxis = 0; tempAxis <= 2; ++tempAxis)
	{
		double tempSplitPoint = nodeAABB.getCenter().get(tempAxis);
		double score = calculateSAHScore(triangles, tempAxis, tempSplitPoint, nodeAABB, buildEntry);

		if (score < lowestScore)
		{
			splitAxis = tempAxis;
			splitPoint = tempSplitPoint;
			lowestScore = score;
		}

		tempSplitPoint = calculateMedianPoint(triangles, tempAxis, buildEntry);
		score = calculateSAHScore(triangles, tempAxis, tempSplitPoint, nodeAABB, buildEntry);

		if (score < lowestScore)
		{
			splitAxis = tempAxis;
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
				score = calculateSAHScore(triangles, tempAxis, tempSplitPoint, nodeAABB, buildEntry);

				if (score < lowestScore)
				{
					splitAxis = tempAxis;
					splitPoint = tempSplitPoint;
					lowestScore = score;
				}
			}
		}
	}
}

double BVH::calculateSAHScore(const std::vector<Triangle>& triangles, uint64_t splitAxis, double splitPoint, const AABB& nodeAABB, const BVHBuildEntry& buildEntry)
{
	assert(buildEntry.end != buildEntry.start);

	AABB leftAABB, rightAABB;
	uint64_t leftCount = 0;
	uint64_t rightCount = 0;

	for (uint64_t i = buildEntry.start; i < buildEntry.end; ++i)
	{
		AABB triangleAABB = triangles[i].getAABB();

		if (triangleAABB.getCenter().get(splitAxis) <= splitPoint)
		{
			leftAABB.expand(triangleAABB);
			leftCount++;
		}
		else
		{
			rightAABB.expand(triangleAABB);
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

double BVH::calculateMedianPoint(const std::vector<Triangle>& triangles, uint64_t splitAxis, const BVHBuildEntry& buildEntry)
{
	std::vector<double> centerPoints;

	for (uint64_t i = buildEntry.start; i < buildEntry.end; ++i)
		centerPoints.push_back(triangles[i].getAABB().getCenter().get(splitAxis));

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
