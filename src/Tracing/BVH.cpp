// Copyright © 2015 Mikko Ronkainen <firstname@mikkoronkainen.com>
// License: MIT, see the LICENSE file.

#include "Precompiled.h"

#include "Tracing/BVH.h"
#include "Tracing/Triangle.h"
#include "Tracing/Ray.h"
#include "Tracing/Intersection.h"

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
